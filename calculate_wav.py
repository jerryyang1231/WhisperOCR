# import os
# import wave
# import contextlib

# # 計算音檔時長並返回秒數
# def get_duration(file_path):
#     with contextlib.closing(wave.open(file_path, 'r')) as f:
#         frames = f.getnframes()
#         rate = f.getframerate()
#         duration = frames / float(rate)
#         return duration

# def format_duration(seconds):
#     mins, secs = divmod(seconds, 60)
#     hours, mins = divmod(mins, 60)
#     return int(hours), int(mins), int(secs)

# # 指定資料夾路徑
# folder_path = '/share/nas169/jerryyang/corpus/mandarin/clean/tr'

# # 初始化總時長為 0
# total_duration = 0.0
# target_duration = 3600  # 目標時長為 3600 秒 (1 小時)
# selected_files = []

# # 遍歷資料夾中的所有檔案並累加時長直到接近目標時長
# for filename in os.listdir(folder_path):
#     if filename.endswith('.wav'):
#         file_path = os.path.join(folder_path, filename)
#         duration = get_duration(file_path)
        
#         # 累加時長並記錄檔名
#         if total_duration + duration <= target_duration:
#             selected_files.append(filename)
#             total_duration += duration

# # 格式化總時長
# hours, minutes, seconds = format_duration(total_duration)

# # 輸出選擇的檔案和總時長
# print(f"選擇的音檔如下（總時長約為 {hours} 小時 {minutes} 分鐘 {seconds} 秒）：")
# for file in selected_files:
#     print(file)

import os
import wave
import contextlib
import shutil

# 計算音檔時長並返回秒數
def get_duration(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def format_duration(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return int(hours), int(mins), int(secs)

# 指定來源資料夾和目標資料夾路徑
source_folder = '/share/nas169/jerryyang/corpus/mandarin/clean/tr'
target_folder = '/share/nas169/jerryyang/corpus/mandarin/clean/tr_subset'

# 確認目標資料夾存在，若不存在則建立
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 初始化總時長為 0
total_duration = 0.0
target_duration = 3600  # 目標時長為 3600 秒 (1 小時)
selected_files = []

# 遍歷來源資料夾中的所有檔案並累加時長直到接近目標時長
for filename in os.listdir(source_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(source_folder, filename)
        duration = get_duration(file_path)
        
        # 累加時長並記錄檔名
        if total_duration + duration <= target_duration:
            selected_files.append(filename)
            total_duration += duration

# 格式化總時長
hours, minutes, seconds = format_duration(total_duration)

# 複製選擇的檔案到目標資料夾
for file in selected_files:
    source_path = os.path.join(source_folder, file)
    target_path = os.path.join(target_folder, file)
    shutil.copyfile(source_path, target_path)

# 輸出選擇的檔案和總時長
print(f"選擇的音檔如下（總時長約為 {hours} 小時 {minutes} 分鐘 {seconds} 秒）：")
for file in selected_files:
    print(file)

