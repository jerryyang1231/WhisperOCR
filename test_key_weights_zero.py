import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers import BeitImageProcessor, BeitModel
from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper

# 自定義初始化函數
def init_key_weights_zero(module):
    if isinstance(module, nn.MultiheadAttention):
        nn.init.constant_(module.in_proj_weight[module.embed_dim:2*module.embed_dim, :], 0)

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.conv3:
            x = self.conv3(x)
        x += x
        return F.relu(x)

class FusionModule(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, device='cuda'):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 定義音頻特徵提取器 
        self.audio_cnn = nn.Sequential(
            ResidualBlock2D(1, 4),
            ResidualBlock2D(4, 8),
            ResidualBlock2D(8, 16),
        ).to(self.device)

        # 定義視覺特徵提取器（使用BEiT模型）
        self.image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-384")
        self.visual_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-384").to(self.device)
        
        # 添加視覺特徵投影層
        self.visual_projection = nn.Linear(768, embed_dim).to(self.device)
        
        # fusion module output 投影到 Whisper log-mel-spec 的頻率
        self.attn_output_2_mels = nn.Linear(embed_dim, 80).to(self.device)

        # Fusion blocks
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ]).to(self.device)
        
         # 添加 layer normalization 層
        self.layer_norms = nn.ModuleList([
            LayerNorm(embed_dim).to(self.device)
            for _ in range(num_layers)
        ])
        
        # 初始化 key 的權重為 0
        self.apply(init_key_weights_zero)
        
    def forward(self, audio, visual):
        # 確保音頻和視覺輸入是 float32 類型
        audio = audio.to(self.device, dtype=torch.float32)
        # visual = visual.to(self.device, dtype=torch.float32)
        
        # 將 Mel 頻譜圖轉換為四維張量
        # [batch_size, channel, num_mels, num_frames]
        audio = audio.unsqueeze(1)
        
        # 使用 torch.autocast 確保模型內部計算都是相同的類型
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            audio_features = self.audio_cnn(audio)

            # Flatten
            audio_features = audio_features.view(audio_features.size(0), -1, audio_features.size(3))
            # print("audio_features' shape :", audio_features.shape)
            audio_features = audio_features.permute(0, 2, 1)
            # print("audio_features' shape :", audio_features.shape)
            # [1, 3000, 1280]
                      
            visual_inputs = self.image_processor(visual,
                                                return_tensors="pt",
                                                do_resize=True, 
                                                size={"height": 384, "width": 384},
            )
            visual_inputs = {k: v.to(self.device) for k, v in visual_inputs.items()}
            visual_outputs = self.visual_model(**visual_inputs)
            visual_features =  visual_outputs.last_hidden_state
            # [1, 577, 768]
            
            # 投影視覺特徵
            visual_features = self.visual_projection(visual_features)
            # print("visual_features' shape :", visual_features.shape)
            # [1, 577, 1280]
            
            # Cross-modal attention
            for i in range(len(self.cross_attention_layers)):
                if i == 0:
                    query = audio_features
                    key = value = visual_features
                    attn_output, _ = self.cross_attention_layers[i](query=query, key=key, value=value)
                    # 第一層用 audio feature extractor 的 output 做 residual
                    attn_output = attn_output + audio_features
                else:
                    query = audio_features
                    key = value = attn_output
                    new_attn_output, _ = self.cross_attention_layers[i](query=query, key=key, value=value)
                    # 後面幾層用 cross attention layer 的 input 做 residual
                    attn_output = new_attn_output + attn_output
                
                # 添加 layer normalization
                attn_output = self.layer_norms[i](attn_output)

        # attn_output = attn_output + audio_features
        attn_output = self.attn_output_2_mels(attn_output)
        
        # 轉成[batch_size, feature_dim, seq_len]
        fused_features = attn_output.permute(0, 2, 1)
                    
        return fused_features

# # 創建 FusionModule 的實例並檢查 key 權重
# embed_dim = 1280
# num_layers = 4
# num_heads = 8
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = FusionModule(embed_dim, num_layers, num_heads, device)

# # 獲取第一層 MultiheadAttention 的 key 權重
# attention_layer = model.cross_attention_layers[0]
# key_weights = attention_layer.in_proj_weight[attention_layer.embed_dim:2*attention_layer.embed_dim, :]

# # 驗證 key 的權重是否全為 0
# print("Key weights are all zeros:", torch.all(key_weights == 0).item())