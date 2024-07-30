# import os
# import wave

# # 定義資料夾路徑
# train_wav_path = "/share/nas169/jerryyang/corpus/aishell3/train/wav"

# # 初始化變數
# total_duration = 0
# target_duration = 3600  # 1 小時的秒數
# final_file = None
# final_folder = None

# # 遍歷資料夾及音檔
# for folder in os.listdir(train_wav_path):
#     folder_path = os.path.join(train_wav_path, folder)
#     if os.path.isdir(folder_path):
#         for wav_file in os.listdir(folder_path):
#             if wav_file.endswith('.wav'):
#                 wav_path = os.path.join(folder_path, wav_file)
#                 # 打開音檔並計算時間
#                 with wave.open(wav_path, 'r') as wf:
#                     frames = wf.getnframes()
#                     rate = wf.getframerate()
#                     duration = frames / float(rate)
#                     total_duration += duration
#                     # 檢查是否已達到目標時間
#                     if total_duration >= target_duration:
#                         final_file = wav_file
#                         final_folder = folder
#                         break
#     if total_duration >= target_duration:
#         break

# if final_file and final_folder:
#     print(f"累計時間達到 1 小時的最後一個檔案是：{final_folder}/{final_file}")
# else:
#     print("累計時間未達 1 小時。")

# import os
# import wave

# # 定義資料夾路徑
# train_wav_path = "/share/nas169/jerryyang/corpus/aishell3/train/wav"

# # 初始化變數
# total_duration_with_SSB0057 = 0
# total_duration_without_SSB0057 = 0
# target_duration = 3600  # 1 小時的秒數
# found_SSB0057 = False

# # 遍歷資料夾及音檔
# for folder in os.listdir(train_wav_path):
#     folder_path = os.path.join(train_wav_path, folder)
#     if os.path.isdir(folder_path):
#         for wav_file in os.listdir(folder_path):
#             if wav_file.endswith('.wav'):
#                 wav_path = os.path.join(folder_path, wav_file)
#                 # 打開音檔並計算時間
#                 with wave.open(wav_path, 'r') as wf:
#                     frames = wf.getnframes()
#                     rate = wf.getframerate()
#                     duration = frames / float(rate)
#                     total_duration_with_SSB0057 += duration
#                     if folder != 'SSB0057':
#                         total_duration_without_SSB0057 += duration
#                     else:
#                         found_SSB0057 = True
#     if found_SSB0057:
#         break

# print(f"加上 SSB0057 的所有音檔的累積時間是：{total_duration_with_SSB0057 / 3600:.2f} 小時")
# print(f"不加 SSB0057 的所有音檔的累積時間是：{total_duration_without_SSB0057 / 3600:.2f} 小時")

# import os
# import wave

# # 定義資料夾路徑
# train_wav_path = "/share/nas169/jerryyang/corpus/aishell3/train/wav"

# # 初始化變數
# total_duration_with_SSB0057 = 0
# total_duration_without_SSB0057 = 0
# target_duration = 3600  # 1 小時的秒數
# found_SSB0057 = False
# folder_list = []

# # 遍歷資料夾及音檔
# for folder in sorted(os.listdir(train_wav_path)):
#     folder_path = os.path.join(train_wav_path, folder)
#     if os.path.isdir(folder_path):
#         folder_list.append(folder)
#         for wav_file in os.listdir(folder_path):
#             if wav_file.endswith('.wav'):
#                 wav_path = os.path.join(folder_path, wav_file)
#                 # 打開音檔並計算時間
#                 with wave.open(wav_path, 'r') as wf:
#                     frames = wf.getnframes()
#                     rate = wf.getframerate()
#                     duration = frames / float(rate)
#                     total_duration_with_SSB0057 += duration
#                     if folder != 'SSB0057':
#                         total_duration_without_SSB0057 += duration
#                     else:
#                         found_SSB0057 = True
#         if found_SSB0057:
#             break

# print(f"到 SSB0057 之前的所有資料夾（包含 SSB0057）：{folder_list}")
# print(f"加上 SSB0057 的所有音檔的累積時間是：{total_duration_with_SSB0057 / 3600:.2f} 小時")
# print(f"不加 SSB0057 的所有音檔的累積時間是：{total_duration_without_SSB0057 / 3600:.2f} 小時")

import os
import wave

def calculate_audio_duration(base_path, target_folder, target_duration=3600):
    # 初始化變數
    total_duration_with_target = 0
    total_duration_without_target = 0
    found_target = False
    folder_list = []

    # 遍歷資料夾及音檔
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            folder_list.append(folder)
            for wav_file in os.listdir(folder_path):
                if wav_file.endswith('.wav'):
                    wav_path = os.path.join(folder_path, wav_file)
                    # 打開音檔並計算時間
                    with wave.open(wav_path, 'r') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                        total_duration_with_target += duration
                        if folder != target_folder:
                            total_duration_without_target += duration
                        else:
                            found_target = True
            if found_target:
                break

    print(f"到 {target_folder} 之前的所有資料夾（包含 {target_folder}）：{folder_list}")
    print(f"加上 {target_folder} 的所有音檔的累積時間是：{total_duration_with_target / 3600:.2f} 小時")
    print(f"不加 {target_folder} 的所有音檔的累積時間是：{total_duration_without_target / 3600:.2f} 小時")

# 使用範例
base_path = "/share/nas169/jerryyang/corpus/aishell3/train/wav"
target_folder = "SSB0339"
calculate_audio_duration(base_path, target_folder)


