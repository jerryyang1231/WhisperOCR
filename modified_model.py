import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BeitImageProcessor, BeitModel
from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return F.relu(x)

class FusionModule(nn.Module):
    def __init__(self, embed_dim=1280, num_layers=12, num_heads=8, device='cuda'):
        super(FusionModule, self).__init__()
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
        
    def forward(self, audio, visual):
        # 確保音頻和視覺輸入是 float32 類型
        audio = audio.to(self.device, dtype=torch.float32)
        visual = visual.to(self.device, dtype=torch.float32)
        
        # 將 Mel 頻譜圖轉換為四維張量
        # [batch_size, channel, num_mels, num_frames]
        audio = audio.unsqueeze(1)
        
        # 使用 torch.autocast 確保模型內部計算都是相同的類型
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            audio_features = self.audio_cnn(audio)

            # Flatten
            audio_features = audio_features.view(audio_features.size(0), -1, audio_features.size(3))
            audio_features = audio_features.permute(0, 2, 1)
            # print("audio_features' shape :", audio_features.shape)
            # [1, 3000, 1280]
                      
            visual_inputs = self.image_processor(visual, return_tensors="pt", do_resize=False)
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
                    attn_output =  attn_output + audio_features
                else:
                    query = audio_features
                    key = value = attn_output
                    new_attn_output, _ = self.cross_attention_layers[i](query=query, key=key, value=value)
                    # 後面幾層用 cross attention layer 的 input 做 residual
                    attn_output = new_attn_output + attn_output
        
        attn_output = self.attn_output_2_mels(attn_output)
        
        # 轉成[batch_size, feature_dim, seq_len]
        fused_features = attn_output.permute(0, 2, 1)
                    
        return fused_features

# 準備輸入數據
batch_size = 1
num_mels = 80
num_frames = 3000  
height = 384
width = 384
# 模擬 Mel 頻譜圖
audio_input = torch.randn(batch_size, num_mels, num_frames)  
# 模擬字幕圖
visual_input = torch.randn(batch_size, 3, height, width)

# 創建FusionModule實例
fusion_module = FusionModule()

# # 通過FusionModule進行特徵提取
fused_features = fusion_module(audio_input, visual_input)
# print("fused_features' shape :", fused_features.shape)

# 將fused_features和tokens移動到同一設備
fused_features = fused_features.to(fusion_module.device)

# 定義模型參數
model_hub = "openai/whisper-base" 
save_path = "/share/nas169/jerryyang/AVfusion/saved_model"  
sampling_rate = 16000

# 創建 Whisper 模型實例
model = Whisper(
    source=model_hub,
    save_path=save_path,
    sampling_rate=sampling_rate,
).to(fusion_module.device)

# 準備輸入數據
tokens = torch.tensor([[1, 1]]) * model.model.config.decoder_start_token_id
tokens = tokens.to(fusion_module.device)

# 使用 FusionModule 的特徵作為 Whisper 模型的輸入
encoder_outputs, decoder_logits, _ = model.forward(fused_features, tokens)
# print("encoder output's shape :",encoder_outputs.shape)
# print("decoder_logits' shape :", decoder_logits.shape)
