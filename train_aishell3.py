#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with librispeech.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on the NLL.

If you want to only use the whisper encoder system, please refer to the recipe
speechbrain/recipes/LibriSpeech/ASR/CTC/train_with_whisper.py

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_hf_whisper.yaml

Authors
 * Adel Moumen 2022, 2024
 * Titouan Parcollet 2022
"""

# my command
# python train_aishell3.py hparams/train_aishell3.yaml --test_only
# python train_aishell3.py hparams/train_aishell3.yaml

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.dataio.dataio import get_image_paths
from speechbrain.augment.time_domain import Resample
import wandb 
from PIL import Image
os.environ["WANDB_DIR"] = "/share/nas169/jerryyang/AVfusion/wandb"

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.Brain):   
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # 創建 Resample 實例
        resampler = Resample(orig_freq=44100, new_freq=16000)
        # 重新採樣
        wavs = resampler(wavs)
        
        # mel_clean's shape = [1, 80, 3000]
        mel_clean = self.modules.whisper._get_mel(wavs)
        mel_clean = mel_clean.to(self.device)

        # id 的列表
        ids = batch.id  # 注意這裡使用ids，而非單一id
        
        # # 使用提取的函數來獲取圖像路徑
        # image_paths = get_image_paths(ids, stage, self.hparams)       
        
        # visual_input = [Image.open(image_path).convert('RGB') for image_path in image_paths]
       
        bos_tokens, bos_tokens_lens = batch.tokens_bos
        
        # # Add waveform augmentation if specified.
        # if hasattr(self.hparams, "wav_augment"):
        #     wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        #     bos_tokens = self.hparams.wav_augment.replicate_labels(bos_tokens)
        #     bos_tokens_lens = self.hparams.wav_augment.replicate_labels(
        #         bos_tokens_lens
        #     )
        
        # # mel_noisy's shape = [1, 80, 3000]
        # mel_noisy = self.modules.whisper._get_mel(wavs)
        # mel_noisy = mel_noisy.to(self.device)
        
        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id
      
        # # 使用 FusionModule 進行特徵融合
        # fused_features = self.modules.fusion_module(mel_noisy, visual_input)
        
        # # training Forward encoder + decoder
        # enc_out_noisy, logits_noisy, _ = self.modules.whisper(fused_features, bos_tokens)
        
        # target generator Forward encoder + decoder
        enc_out_clean, logits_clean, _ = self.modules.whisper(mel_clean, bos_tokens)
        log_probs_clean = self.hparams.log_softmax(logits_clean)
        
        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out_clean.detach(), wav_lens
            )
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(
                enc_out_clean.detach(), wav_lens
            )

        return log_probs_clean, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        (log_probs_clean, hyps, wav_lens) = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        
        loss = self.hparams.nll_loss(
            log_probs_clean, tokens_eos, length=tokens_eos_lens
        )
        
        # # Label Augmentation
        # if hasattr(self.hparams, "wav_augment"):
        #     tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
        #     tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
        #         tokens_eos_lens
        #     )
        
        # loss_asr = self.hparams.nll_loss(
        #     log_probs_clean, tokens_eos, length=tokens_eos_lens
        # )
        # self.ob_loss_asr = loss_asr
        
        # loss_mel = self.hparams.l1_loss(fused_features, mel_clean)
        # self.ob_loss_mel = loss_mel
        
        # loss_enc = self.hparams.l1_loss(enc_out_noisy, enc_out_clean)
        # self.ob_loss_enc = loss_enc
        
        # # 修改 logits_clean 以進行 one-hot 編碼並計算 target class
        # target_class = torch.argmax(logits_clean, dim=-1)
        
        # # 將 logits_noisy 轉換為 [batch_size * seq_len, num_classes] 的形狀
        # logits_noisy_flat = logits_noisy.view(-1, logits_noisy.size(2))
        
        # # 將 target_class 轉換為 [batch_size * seq_len] 的形狀
        # target_class_flat = target_class.view(-1)
        
        # # 計算 cross entropy loss
        # loss_dec = F.cross_entropy(logits_noisy_flat, target_class_flat)
        # self.ob_loss_dec = loss_dec
        
        # alpha = 1
        # beta = 1
        # gamma = 1
        # delta = 1
        
        # loss = alpha * loss_asr + beta * loss_mel + gamma * loss_enc + delta * loss_dec
        # self.ob_loss = loss
        
        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens             
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in hyps
            ]
            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(target_words, skip_special_tokens=True)
            
            predicted_words = [text.split(" ") for text in predicted_words]
            target_words = [text.split(" ") for text in target_words]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Log to WandB
        if if_main_process():  # Only log once per epoch
            if stage == sb.Stage.TRAIN:
                wandb.log({"train_loss": stage_loss,
                })
            elif stage == sb.Stage.VALID:
                wandb.log({
                    "valid_loss": stage_loss,
                    "valid_CER": stage_stats["CER"],
                })
            elif stage == sb.Stage.TEST:
                wandb.log({
                    "test_loss": stage_loss,
                    "test_CER": stage_stats["CER"],
                })
        
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # 改成留CER最低的
            self.checkpointer.save_and_keep_only(
                meta={"CER": stage_stats["CER"]},
                min_keys=["CER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
        
    def on_fit_batch_end(self, batch, outputs, loss, should_step):     
        
        # # Log to WandB
        # if if_main_process():  # Only log once per step
        #     wandb.log({
        #         "batch_train_total_loss": self.ob_loss.item(),
        #         "batch_train_loss_mel": self.ob_loss_mel.item(),
        #         "batch_train_loss_enc": self.ob_loss_enc.item(),
        #         "batch_train_loss_dec": self.ob_loss_dec.item(),
        #         "batch_train_loss_asr": self.ob_loss_asr.item(),
        #     })
        
        # Log to WandB
        if if_main_process():  # Only log once per step
            wandb.log({
                "batch_train_loss": loss.item(),
            })

def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )
    
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        # name = 檔案名稱
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file,
            replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig
    
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    
    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    # Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )
    
    return train_data, valid_data, test_datasets

if __name__ == "__main__":
    
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Initialize WandB
    wandb.init(project="v6", 
               config=hparams,
               name="finetune whisper on reorganize aishell3 dataset ",
    )
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing aishell3)
    from aishell3_prepare import prepare_aishell3  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell3,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_train_lst": hparams["train_splits"],
            "merge_train_name": "train.csv",
            "merge_valid_lst": hparams["dev_splits"],
            "merge_valid_name": "valid.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer
    
    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)
    
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["whisper_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # # We load the pretrained whisper model
    # if "pretrainer" in hparams.keys():
    #     run_on_main(hparams["pretrainer"].collect_files)
    #     hparams["pretrainer"].load_collected()
    
    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer
    
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )
    
    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="CER",
        )
