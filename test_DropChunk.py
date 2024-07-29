import torch
import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.augment.time_domain import DropChunk

# 加載音頻文件
signal = read_audio('/share/nas169/jerryyang/speechbrain/tests/samples/single-mic/example1.wav')

# # 創建 DropChunk 對象
# dropper = DropChunk(
#     drop_length_low=2000,
#     drop_length_high=4000,
#     drop_count_low=1,
#     drop_count_high=10,
# )

# 創建 DropChunk 對象
dropper = DropChunk(
    drop_length_low=1,
    drop_length_high=5,
    drop_count_low=10,
    drop_count_high=20,
)


# 添加 batch 維度
signal = signal.unsqueeze(0)  # [batch, time]

# 設置相對長度
length = torch.ones(1)

# 應用 DropChunk
dropped_signal = dropper(signal, length)

# 移除 batch 維度
dropped_signal = dropped_signal.squeeze(0)  # [time]

# 確保音頻張量的形狀為 [channels, time]
if dropped_signal.ndim == 1:
    dropped_signal = dropped_signal.unsqueeze(0)  # [1, time]

# 設置音頻文件的保存路徑
output_file = '/share/nas169/jerryyang/speechbrain/tests/samples/single-mic/dropped_signal_2.wav'

# 保存音頻文件，使用原始音頻的取樣率
sample_rate = 16000  # 原始音頻的取樣率
torchaudio.save(output_file, dropped_signal, sample_rate=sample_rate)
print(f"音頻文件已保存到: {output_file}")
