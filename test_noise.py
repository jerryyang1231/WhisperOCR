import pytest
import torch
from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.augment.time_domain import AddNoise

# 讀取乾淨的語音信號
# signal = read_audio('/share/nas169/jerryyang/speechbrain/tests/samples/single-mic/example1.wav')
signal = read_audio('/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050001.wav')
clean = signal.unsqueeze(0)  # [batch, time, channels]

# 創建加噪音的實例
noisifier = AddNoise('/share/nas169/jerryyang/speechbrain/tests/samples/annotation/noise.csv',
                    replacements={'noise_folder': '/share/nas169/jerryyang/speechbrain/tests/samples/noise'},
                    noise_sample_rate=44100,
                    clean_sample_rate=44100,
                    snr_low=10,  
                    snr_high=20,
    )

# 添加噪音
noisy = noisifier(clean, torch.ones(1))

# 將加了噪音的語音信號保存到檔案中
output_file = '/share/nas169/jerryyang/corpus/mandarin/noisy/noisy_example1.wav'
write_audio(output_file, noisy.squeeze(0).cpu(), 44100)

# import pytest
# import torch
# from speechbrain.dataio.dataio import read_audio, write_audio
# from speechbrain.augment.time_domain import AddNoise

# # 定義乾淨的語音信號檔案路徑
# clean_files = [
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050001.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050002.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050003.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050004.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050005.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050006.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050007.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050008.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050009.wav',
#     '/share/nas169/jerryyang/corpus/mandarin/clean/tr/SSB00050010.wav',
# ]

# # 創建加噪音的實例
# noisifier = AddNoise(
#     '/share/nas169/jerryyang/speechbrain/tests/samples/annotation/noise.csv',
#     replacements={'noise_folder': '/share/nas169/jerryyang/speechbrain/tests/samples/noise'},
#     noise_sample_rate=44100,
#     clean_sample_rate=44100,
#     snr_low=10,
#     snr_high=20,
# )

# # 逐個讀取乾淨的語音信號並添加噪音
# for i, clean_file in enumerate(clean_files):
#     # 讀取乾淨的語音信號
#     signal = read_audio(clean_file)
#     clean = signal.unsqueeze(0)  # [batch, time, channels]

#     # 添加噪音
#     noisy = noisifier(clean, torch.ones(1))

#     # 將加了噪音的語音信號保存到檔案中
#     output_file = f'/share/nas169/jerryyang/corpus/mandarin/noisy/noisy_example_{i + 1}.wav'
#     write_audio(output_file, noisy.squeeze(0).cpu(), 44100)


