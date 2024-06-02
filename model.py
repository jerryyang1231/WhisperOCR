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
    def __init__(self, audio_embed_dim=80, visual_embed_dim=80, num_heads=8, num_layers=6, num_mels=80, num_frames=1000):
        super(FusionModule, self).__init__()
        self.audio_embed_dim = audio_embed_dim
        self.visual_embed_dim = visual_embed_dim
        self.num_mels = num_mels
        
        # 定義音頻特徵提取器 (3層ResidualBlock2D)
        self.audio_cnn = nn.Sequential(
            ResidualBlock2D(1, 64),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock2D(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock2D(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        
        # 計算展平後的大小
        sample_input = torch.randn(1, 1, num_mels, num_frames)
        sample_output = self.audio_cnn(sample_input)
        flattened_size = sample_output.view(1, -1).size(1)
        
        # 線性層處理音頻特徵
        self.audio_linear = nn.Linear(flattened_size, audio_embed_dim)

        # 定義視覺特徵提取器
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Initial Conv layer for visual input
            ResidualBlock2D(64, 128),
            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),  # Conv. Transpose Layer
            ResidualBlock2D(256, 384),
            nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2),  # Conv. Transpose Layer
            ResidualBlock2D(384, 384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # Final Conv layer for visual input
            nn.AdaptiveAvgPool2d((1, 1))  # Average Pooling Layer
        )
        
        # 計算展平後的大小
        video_sample_input = torch.randn(1, 3, 224, 224)  # 減少圖像尺寸
        video_sample_output = self.visual_cnn(video_sample_input)
        video_flattened_size = video_sample_output.view(1, -1).size(1)
        
        # 線性層處理視覺特徵
        self.visual_linear = nn.Linear(video_flattened_size, visual_embed_dim)

        # 定義音頻和視覺特徵的線性層
        self.audio_to_embed = nn.Linear(audio_embed_dim, audio_embed_dim)
        self.visual_to_embed = nn.Linear(visual_embed_dim, visual_embed_dim)

        # Fusion blocks
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=audio_embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(audio_embed_dim, audio_embed_dim)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(audio_embed_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, audio, visual):
        # 將 Mel 頻譜圖轉換為四維張量
        audio = audio.unsqueeze(1)  # 添加通道維度，使其形狀為 [batch_size, 1, num_mels, num_frames]
        audio_features = self.audio_cnn(audio)
        audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten
        audio_features = self.audio_linear(audio_features)
        
        # 提取視覺特徵
        visual_features = self.visual_cnn(visual)
        visual_features = visual_features.view(visual_features.size(0), -1)  # Flatten
        visual_features = self.visual_linear(visual_features)
        
        # Cross-modal attention
        for i in range(len(self.cross_attention_layers)):
            if i == 0:
                audio_features = self.audio_to_embed(audio_features)  # 經過線性層
                visual_features = self.visual_to_embed(visual_features)  # 經過線性層
                query = visual_features.unsqueeze(0)  # 形狀為 [1, 1, 80]
                key = value = audio_features.unsqueeze(0)  # 形狀為 [1, 1, 80]
            else:
                attn_output = self.audio_to_embed(attn_output)  # 經過線性層
                visual_features = self.visual_to_embed(visual_features)  # 經過線性層
                query = visual_features.unsqueeze(0)  # 形狀為 [1, 1, 80]
                key = value = attn_output
                
            attn_output, _ = self.cross_attention_layers[i](query=query, key=key, value=value)
            attn_output = self.fc_layers[i](attn_output)
            attn_output = self.layer_norms[i](attn_output + query)  # Add & Norm

        fused_features = attn_output.permute(1, 0, 2)  # (batch, seq_len, feature_dim)
        
        # 确保输出形状符合 Whsiper 的输入要求
        fused_features = fused_features.permute(0, 2, 1)  # 转换为 [batch_size, feature_dim, seq_len]
        
        return fused_features

# 創建FusionModule實例
num_mels = 80
num_frames = 1000  # 減少音頻幀數
fusion_module = FusionModule(num_mels=num_mels, num_frames=num_frames)

# 準備輸入數據
batch_size = 1
audio_input = torch.randn(batch_size, num_mels, num_frames)  # 模擬 Mel 頻譜圖
visual_input = torch.randn(batch_size, 3, 224, 224)  # 減少圖像尺寸

# 通過FusionModule進行特徵提取
fused_features = fusion_module(audio_input, visual_input)
# print("fused_features :",fused_features)

# 打印輸入和輸出形狀
print("audio_input shape :", audio_input.shape)
print("visual_input shape :", visual_input.shape)
print("fused_features shape :", fused_features.shape)

# 定义模型参数
model_hub = "openai/whisper-tiny"  # 这里你可以选择其他模型，如 "whisper-base", "whisper-large", 等等。
save_path = "/share/nas169/jerryyang/AVfusion/saved_model"  # 指定模型保存的路径
sampling_rate = 16000

# 创建 Whisper 模型实例
model = Whisper(
    source=model_hub,
    save_path=save_path,
    sampling_rate=sampling_rate,
)

# 准备输入数据
tokens = torch.tensor([[1, 1]]) * model.model.config.decoder_start_token_id

# 使用 Whisper 模型中的 pad_or_trim 函数来填充或截断特征
fused_features_padded = model.pad_or_trim(fused_features, length=3000)
# print("fused_features_padded :", fused_features_padded)
print("fused_features_padded shape :",fused_features_padded.shape)

# 使用 FusionModule 的特征作为 Whisper 模型的输入
encoder_outputs, decoder_logits, decoder_attn = model.forward(fused_features_padded, tokens)
print("encoder output's shape :",encoder_outputs.shape)
print("decoder_logits' shape :", decoder_logits.shape)
if decoder_attn:
    print("decoder_attn's shape :", decoder_attn.shape)