import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import pandas as pd
import whisper
import evaluate
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    load_librispeech_data,  # 使用專門為 LibriSpeech 設計的數據加載函數
    load_wave,
    add_noise,
    WhisperDataCollatorWhithPadding,
    whisper_optimizer,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import LengthBatchSampler
import soundfile as sf

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

# my command
# python -u whisper_ft_librispeech.py config/audio/audio_en-x_tiny.yaml

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        print("Dataloader max length : {}".format(max_length))
        print("Loaded {} noise wavs".format(len(self.noise_fn)))

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        lang = 'en'
        audio_path, text, _ = self.audio_info_list[id]  # LibriSpeech 使用音頻和文本的二元組
        wav_data, sample_rate = sf.read(audio_path)  # 使用 soundfile 讀取 FLAC 檔案
        
        if np.random.rand() > self.noise_prob: # disable noise
            audio = wav_data.flatten().astype(np.float32) / 32768.0
        else: # add noise
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32) / 32768.0

        audio_frames = len(audio.flatten()) // 160
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels) 
    
        if self.spec_augment:
            if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T
            elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T
            else:
                raise NotImplementedError 

        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

class WhisperModelModule(LightningModule):
    def __init__(self, cfg, model_name, lang, train_dataset, val_dataset, test_dataset) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name, 
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/whisper-flamingo/experiments/whisper',
                                        dropout_rate=cfg.dropout_rate)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, task='transcribe')

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.train_dataset = train_dataset
        self.__val_dataset = val_dataset
        self.__test_dataset = test_dataset
        self.special_token_set = set(self.tokenizer.special_tokens.values())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features, x_v = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_id, dataloader_idx=None):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features, x_v = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        labels[labels == -100] = self.tokenizer.eot
        tokens = torch.argmax(out, dim=2)

        eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))
        first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).cuda() * eot_find, dim=1, keepdim=True)
        tokens[torch.arange(eot_find.shape[1]).cuda() > first_eot] = self.tokenizer.eot

        mask = ~(tokens[:, 3:] == self.tokenizer.eot)
        n_correct = torch.sum(
            tokens[:, 3:].masked_select(mask).eq(labels[:, 3:].masked_select(mask))
        )
        total = torch.sum(mask)
        acc = n_correct.item() / (total.item() + 1e-8)
        acc = acc if acc < 1 else 0

        o_list, o_list_full, l_list, l_list_full = [], [], [], []
        for o, l in zip(tokens, labels):
            o_list.append(self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set]))
            l_list.append(self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set]))
        wer, cer = wer_cer(hypo=o_list, ref=l_list)
        
        for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
            print("-"*10)
            print("PRED: {}".format(hypo))
            print("REF:  {}".format(ref))
            if i == 1: break
        
        # dataloader_idx抓不到，會報錯
        log_prefix = {0: 'dev-clean', 1: 'dev-other', 2: 'test-clean', 3: 'test-other'}
        self.log("{}/loss".format(log_prefix[dataloader_idx]), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/cer".format(log_prefix[dataloader_idx]), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer".format(log_prefix[dataloader_idx]), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc".format(log_prefix[dataloader_idx]), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        
        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total, video=False)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = LibriSpeechDataset(self.__train_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=self.cfg.spec_augment,
                                      noise_prob=cfg.noise_prob,
                                    #   noise_fn=cfg.noise_fn,
                                    )   
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * self.cfg.batch_size,
                    shapes=[i[2] for i in self.__train_dataset], 
                    sort_in_batch='descending',
                    sort_batch='shuffle',
                    drop_last=True,)
        if cfg.num_devices > 1:
            print("Using distributed sampler")
            length_sorter = DistributedSamplerWrapper(length_sorter)
        return torch.utils.data.DataLoader(dataset,
                        batch_sampler=length_sorter,
                        num_workers=self.cfg.num_worker,
                        collate_fn=WhisperDataCollatorWhithPadding())

    def val_dataloader_clean(self):
        dataset = LibriSpeechDataset(self.__val_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=0
                                    )
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[2] for i in self.__val_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
    
    def val_dataloader_noisy(self):
        dataset = LibriSpeechDataset(self.__val_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=1,
                                      noise_fn=cfg.noise_fn_val,
                                    )
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[2] for i in self.__val_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
    def test_dataloader_clean(self):
        dataset = LibriSpeechDataset(self.__test_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=0
                                    )
        
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[2] for i in self.__test_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
    
    def test_dataloader_noisy(self):
        dataset = LibriSpeechDataset(self.__test_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=1,
                                      noise_fn=cfg.noise_fn_test,
                                    )
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[2] for i in self.__test_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

cfg_yaml = sys.argv[1]
with open(cfg_yaml, 'r') as file:
    dct = yaml.safe_load(file)
    cfg = types.SimpleNamespace(**dct)

print(cfg)
print("audio max length: {}".format(cfg.audio_max_length))

tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                            cfg.check_output_dir, 
                                                                            cfg.train_name, 
                                                                            cfg.train_id,
                                                                            cfg.monitor,)

# 使用 LibriSpeech 數據集
audio_transcript_pair_list = load_librispeech_data(cfg.audio_max_length, cfg.text_max_length, include_audio_lens=True)

model = WhisperModelModule(cfg, cfg.model_name, cfg.lang, audio_transcript_pair_list['train'], 
                                                          audio_transcript_pair_list['valid'],
                                                          audio_transcript_pair_list['test'])

trainer = Trainer(
    precision=cfg.precision,
    accelerator="gpu",
    max_steps=cfg.num_train_steps,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list,
    num_sanity_val_steps=0, # default is 2 batches, 0 to turn off
    devices=cfg.num_devices,
    val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps), # validate after this number batches
    check_val_every_n_epoch=None, # If None, validation will be done solely based on the number of training batches
    reload_dataloaders_every_n_epochs=1, # shuffle the dataloader after an epoch
    use_distributed_sampler=False, # implemented custom distributed trainer
)

print(cfg)
resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
if os.path.exists(resume_ckpt) and cfg.resume_training: # resume training, don't validate
    trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader_clean(), model.test_dataloader_clean()])
else:
    trainer.validate(model=model, dataloaders=[model.val_dataloader_clean()]) # validate before training
    # trainer.fit(model, val_dataloaders=[model.val_dataloader_clean()])
