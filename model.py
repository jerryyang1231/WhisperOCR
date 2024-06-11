import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper

# (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
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
    def __init__(self, embed_dim=1152, num_layers=12):
        super(FusionModule, self).__init__()
        self.embed_dim = embed_dim
        
        # 定義音頻特徵提取器 
        self.audio_cnn = nn.Sequential(
            ResidualBlock2D(1, 4),
            ResidualBlock2D(4, 8),
            ResidualBlock2D(8, 16),
        )

        # 定義視覺特徵提取器
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # Initial Conv layer for visual input
            ResidualBlock2D(8, 16),
            ResidualBlock2D(16, 32),
            ResidualBlock2D(32, 64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Final Conv layer for visual input
            nn.AdaptiveAvgPool2d((3, 3))  # Average Pooling Layer
        )
        
        # 添加音頻特徵投影層
        self.audio_projection = nn.Linear(1280, embed_dim)
        
        # fusion module output 投影到 Whisper log-mel-spec 的頻率
        self.attn_output_2_mels = nn.Linear(embed_dim, 80)

        # Fusion blocks
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, batch_first=True)
            for _ in range(num_layers)
        ])
        
    def forward(self, audio, visual):
        # 將 Mel 頻譜圖轉換為四維張量
        # 添加通道維度，使其形狀為 [batch_size, 1, num_mels, num_frames]
        audio = audio.unsqueeze(1)
        audio_features = self.audio_cnn(audio) 
        # Flatten
        # [batch_size, 1*num_mels, num_frames]
        audio_features = audio_features.view(audio_features.size(0), -1, audio_features.size(3))
        # [batch_size, seq_len, feature_dim]
        audio_features = audio_features.permute(0, 2, 1)
        # print("audio_features' shape :", audio_features.shape)
                
        # 投影音頻特徵
        # audio 配合 visual 的特徵維度
        audio_features = self.audio_projection(audio_features)
        # print("projected audio_features' shape:", audio_features.shape)

        # 提取視覺特徵
        # [batch_size, 3, height, width]
        visual_features = self.visual_cnn(visual)
        # Flatten
        # [batch_size, feature_dim]
        visual_features = visual_features.view(visual_features.size(0), -1)
        # [batch_size, 1, feature_dim]
        visual_features = visual_features.unsqueeze(1)
        # print("visual_features' shape :", visual_features.shape)
        
        # 複製 visual_features 以匹配 audio_features 的序列長度
        visual_features = visual_features.expand(visual_features.size(0), audio_features.size(1), visual_features.size(2))
        # print("expanded visual_features' shape:", visual_features.shape)
        
        # Cross-modal attention
        for i in range(len(self.cross_attention_layers)):
            if i == 0:
                query = visual_features
                key = value = audio_features
            else:
                query = visual_features
                key = value = attn_output
                
            attn_output, _ = self.cross_attention_layers[i](query=query, key=key, value=value)
            # attn_output = self.fc_layers[i](attn_output)
            # attn_output = self.layer_norms[i](attn_output + query)  # Add & Norm

        attn_output = self.attn_output_2_mels(attn_output)
        # print("attn_output's shape:", attn_output.shape)
        
        # 轉成[batch_size, feature_dim, seq_len]
        # 符合log-mel-spec 的形狀
        fused_features = attn_output.permute(0, 2, 1)
        # print("fused_features's shape:", fused_features.shape)
        
        return fused_features

import torchvision.transforms as transforms
from PIL import Image
import torch

def read_image(image_path):
    """
    讀取圖片並進行預處理，將其轉換為張量。

    參數
    ----
    image_path : str
        圖片的文件路徑。

    返回
    ----
    torch.Tensor
        經過預處理的圖片張量。
    """
    # 定義圖片轉換
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # 調整圖片大小
        transforms.ToTensor(),          # 轉換為張量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ])

    # 打開圖片
    image = Image.open(image_path).convert('RGB')
    
    # 應用轉換
    image_tensor = transform(image)
    
    return image_tensor

##################################################################
# 模擬數據測試
##################################################################

# # 準備輸入數據
# batch_size = 1
# num_mels = 80
# num_frames = 3000  
# height = 384
# width = 384
# # 模擬 Mel 頻譜圖
# audio_input = torch.randn(batch_size, num_mels, num_frames)  
# # 模擬字幕圖
# visual_input = torch.randn(batch_size, 3, height, width) 

# # 創建FusionModule實例
# fusion_module = FusionModule()

# # 通過FusionModule進行特徵提取
# fused_features = fusion_module(audio_input, visual_input)
# # print("fused_features :", fused_features)

# # # 打印輸入和輸出形狀
# # print("audio_input shape :", audio_input.shape)
# # print("visual_input shape :", visual_input.shape)
# # print("fused_features shape :", fused_features.shape)

# # 定義模型參數
# model_hub = "openai/whisper-tiny" 
# save_path = "/share/nas169/jerryyang/AVfusion/saved_model"  
# sampling_rate = 16000

# # 創建 Whisper 模型實例
# model = Whisper(
#     source=model_hub,
#     save_path=save_path,
#     sampling_rate=sampling_rate,
# )

# # 準備輸入數據
# tokens = torch.tensor([[1, 1]]) * model.model.config.decoder_start_token_id

# # 使用 FusionModule 的特徵作為 Whisper 模型的輸入
# encoder_outputs, decoder_logits, decoder_attn = model.forward(fused_features, tokens)
# # print("encoder output's shape :",encoder_outputs.shape)
# # print("decoder_logits' shape :", decoder_logits.shape)
# # if decoder_attn:
# #     print("decoder_attn's shape :", decoder_attn.shape)
