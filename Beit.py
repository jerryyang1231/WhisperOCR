# from transformers import BeitImageProcessor, BeitForMaskedImageModeling
# import torch
# import torch.nn as nn
# from PIL import Image
# import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-384")
# model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-384")

# model.beit.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, model.config.hidden_size))

# num_patches = (model.config.image_size // model.config.patch_size) ** 2
# pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
# # create random boolean mask of shape (batch_size, num_patches)
# bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

# outputs = model(pixel_values, bool_masked_pos=bool_masked_pos, output_hidden_states=True)
# loss, logits = outputs.loss, outputs.logits
# list(logits.shape)

# hidden_states = outputs.hidden_states
# last_hidden_state = hidden_states[-1]
# print("Shape of last hidden state:", last_hidden_state.shape)

from transformers import BeitImageProcessor, BeitModel
import torch
from datasets import load_dataset

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# print("image's size:", image.size)  # (width, height)

batch_size = 1 
height = 384
width = 384
# 模擬字幕圖
visual_input = torch.randn(batch_size, 3, height, width)

image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-384")
model = BeitModel.from_pretrained("microsoft/beit-base-patch16-384")

inputs = image_processor(visual_input, return_tensors="pt", do_resize=False)
# print("inputs' pixel_values :", inputs['pixel_values'])

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)
print("last_hidden_states' shape :", last_hidden_states.shape)
