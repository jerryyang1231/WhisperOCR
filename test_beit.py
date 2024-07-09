from transformers import BeitImageProcessor
from PIL import Image
import torch

# 初始化 BeitImageProcessor
processor = BeitImageProcessor(
    do_resize=True, 
    size={"height": 256, "width": 256},
    do_center_crop=True, 
    crop_size={"height": 224, "width": 224},
    rescale_factor=1/255, 
    do_rescale=True,
    do_normalize=True, 
    image_mean=[0.485, 0.456, 0.406], 
    image_std=[0.229, 0.224, 0.225]
)

# 讀取圖片
# /share/nas169/jerryyang/corpus/mandarin/image/b_1/cv/SSB07370296.jpg
image_path = '/share/nas169/jerryyang/corpus/mandarin/image/b_1/cv/SSB07370296.jpg'
image = Image.open(image_path).convert('RGB')

# 預處理圖片
inputs = processor(images=image, return_tensors="pt")  # 將圖片轉換為 PyTorch tensor

print(inputs['pixel_values'].shape)

# from transformers import BeitImageProcessor
# import torch

# # 初始化 BeitImageProcessor
# processor = BeitImageProcessor(
#     do_resize=True, 
#     size={"height": 256, "width": 256},
#     do_center_crop=True, 
#     crop_size={"height": 224, "width": 224},
#     rescale_factor=1/255, 
#     do_rescale=True,
#     do_normalize=True, 
#     image_mean=[0.485, 0.456, 0.406], 
#     image_std=[0.229, 0.224, 0.225]
# )

# # 假設 image_tensor 是一個已經存在的張量，且值範圍在 [0, 1]
# # 如果值範圍在 [0, 255]，請將 do_rescale 設置為 False
# image_tensor = torch.rand(3, 384, 384)  # 這是一個示例張量

# # 預處理圖片張量
# inputs = processor(images=image_tensor, return_tensors="pt", do_rescale=False)  # 已經是張量，不需要再 rescale

# print(inputs['pixel_values'].shape)