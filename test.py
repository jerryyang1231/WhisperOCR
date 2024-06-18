
import torch

from speechbrain.lobes.models.huggingface_transformers import Whisper

model_hub = "openai/whisper-tiny"
save_path = "savedir"
sampling_rate = 16000
model = Whisper(model_hub, save_path, sampling_rate, language="chinese")
tokens = torch.tensor([[1, 1]]) * model.model.config.decoder_start_token_id
inputs = torch.randn([1, 93680])
mel = model._get_mel(inputs)
outputs = model(mel, tokens)

language = model.language
print("language :", language)

