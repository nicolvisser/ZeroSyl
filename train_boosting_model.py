import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import tgt
import torch
import wandb
from torch import amp, nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchcodec.decoders import AudioDecoder
from tqdm.autonotebook import tqdm

from zerosyl.wavlm import WavLM, WavLMConfig


# fmt: off
@dataclass
class TrainConfig:
    # --- Wandb details ---
    entity: str = "zerospeech"
    project: str = "zerosyl-boost-discrete"
    name: str = "layer-11-window-3-prominence-0.5-k-10000"

    # --- General training control ---
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    accumulation_steps: int = 6
    grad_clip_max_norm: float = 10.0
    batch_size: int = 16
    num_workers: int = 16
    max_duration: float = 15.62
    normalize: bool = True

    # --- Data configuration ---
    train_wav_dir: str = "/home/nicolvisser/Data/waveforms/LibriSpeech"
    train_teacher_dir: str = "/home/nicolvisser/Data/zerosyl/layer-11-window-3-prominence-0.5/tokens-and-lengths"
    train_wav_pattern: str = "train*/**/*.flac"
    train_teacher_pattern: str = "train*/**/*.pt"

    valid_wav_dir: str = "/home/nicolvisser/Data/waveforms/LibriSpeech"
    valid_teacher_dir: str = "/home/nicolvisser/Data/zerosyl/layer-11-window-3-prominence-0.5/tokens-and-lengths"
    valid_wav_pattern: str = "dev*/**/*.flac"
    valid_teacher_pattern: str = "dev*/**/*.pt"

    # --- Model configuration ---
    target_vocab_size: int = 10000
    warmstart_checkpoint_path: str = "checkpoints/WavLM-Large.pt"

    # --- Optimizer / learning rate schedule ---
    lr_init: float = 1e-7
    lr_max: float = 2e-5
    lr_final: float = 2e-5
    n_linear_steps: int = 1000
    n_decay_steps: int = 24000

# fmt: ons


class WavLMWithPredictionHead(nn.Module):
    def __init__(self, wavlm_cfg: WavLMConfig, train_cfg: TrainConfig):
        super().__init__()
        self.wavlm = WavLM(wavlm_cfg)
        self.proj = nn.Linear(wavlm_cfg.encoder_embed_dim, train_cfg.target_vocab_size)

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, mask), padding_mask = self.wavlm.extract_features(
            source=source,
            padding_mask=padding_mask,
            mask=mask,
            ret_mask_indices=True,
        )
        logits = self.proj(x)
        return logits, mask

    @classmethod
    def warmstart_from_pretrained_checkpoint(
        self, train_cfg: TrainConfig, device: torch.device = "cpu"
    ) -> "WavLMWithPredictionHead":
        checkpoint = torch.load(train_cfg.warmstart_checkpoint_path)
        wavlm_cfg = WavLMConfig(checkpoint["cfg"])
        model = WavLMWithPredictionHead(wavlm_cfg, train_cfg)
        model.wavlm.load_state_dict(checkpoint["model"])
        return model.to(device)


class WaveformWithFramewiseLabelsDataset(Dataset):
    def __init__(
        self,
        wav_dir: str,
        teacher_dir: str,
        wav_pattern: str,
        teacher_pattern: str,
        max_duration: float = float("inf"),
        normalize: bool = True,
    ):
        assert (
            max_duration / 0.02 % 1 == 0
        ) or max_duration == math.inf, "max_duration must be a multiple of 0.02 seconds"
        self.max_duration = max_duration

        self.normalize = normalize

        self.wav_paths = {p.stem: p for p in Path(wav_dir).glob(wav_pattern)}
        assert len(self.wav_paths) > 0
        self.teacher_paths = {p.stem: p for p in Path(teacher_dir).glob(teacher_pattern)}
        assert len(self.teacher_paths) > 0
        msg = "waveform and label files must have the same stems in their file names"
        assert set(self.wav_paths.keys()) == set(self.teacher_paths.keys()), msg
        self.stems = sorted(list(self.wav_paths.keys()))

    @property
    def max_duration_frames(self):
        return int(self.max_duration // 0.02)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        # get paths to data
        stem = self.stems[idx]
        wav_path = self.wav_paths[stem]
        teacher_path = self.teacher_paths[stem]

        # get duration of waveform from audio decoder
        decoder = AudioDecoder(wav_path, sample_rate=16000, num_channels=1)
        available_frames = int(decoder.metadata.duration_seconds_from_header // 0.02)

        # crop with a randomly offset window if too long
        extra_frames = max(0, available_frames - self.max_duration_frames)
        start_frame = random.randrange(0, extra_frames, 1) if extra_frames > 0 else 0
        stop_frame = min(available_frames, start_frame + self.max_duration_frames)
        duration_frames = stop_frame - start_frame

        # load the samples
        audio = decoder.get_samples_played_in_range(
            start_seconds=start_frame * 0.02,
            stop_seconds=stop_frame * 0.02,
        )

        if self.normalize:
            wav = torch.nn.functional.layer_norm(audio.data, audio.data.shape)

        # pad such that output of model will give exactly `duration_frames` frames
        wav = torch.nn.functional.pad(
            wav, ((400 - 320) // 2, (400 - 320) // 2)
        ).squeeze(0)

        # load the teacher data
        teacher_data = torch.load(teacher_path)
        assert isinstance(teacher_data, dict)
        labels = torch.repeat_interleave(teacher_data['tokens'], teacher_data['lengths'], dim=0)
        assert labels.ndim == 1

        if len(labels) != available_frames:
            if abs(len(labels) - available_frames) > 2:
                raise ValueError(
                    f"too big of a mismatch between lengths of labels and expected features. check data"
                )
            if len(labels) < available_frames:
                print(len(labels), available_frames)
                raise ValueError(
                    "the teacher data should at least be as long as the expected feature lengths. check data"
                )
            # otherwise just drop the last teacher frame or two. no biggy.
            
        labels = labels[start_frame:stop_frame]

        return wav, labels, duration_frames

    def collate(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        wavs, labels, seqlens = zip(*batch)
        wavs = pad_sequence(wavs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)
        seqlens = torch.tensor(seqlens)
        return wavs, labels, seqlens


class LinearRampCosineDecayScheduler(LRScheduler):
    """
    Custom learning rate scheduler that increases linearly for n_linear_steps,
    then decays with cosine annealing for n_decay_steps,
    then stays at lr_final for the remaining steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        n_linear_steps (int): Number of steps for linear increase.
        n_decay_steps (int): Number of steps for cosine decay.
        lr_init (float, optional): Initial learning rate. Default is 0.
        lr_max (float, optional): Maximum learning rate. Default is 1e-5.
        lr_final (float, optional): Final learning rate. Default is 1e-6.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_linear_steps: int,
        n_decay_steps: int,
        lr_init: float,
        lr_max: float,
        lr_final: float,
    ):
        self.n_linear_steps = n_linear_steps
        self.n_decay_steps = n_decay_steps

        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final

        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        current_step = self.last_epoch

        if current_step <= self.n_linear_steps:
            lr = self.lr_init + (self.lr_max - self.lr_init) * current_step / (
                self.n_linear_steps
            )
        elif current_step <= self.n_linear_steps + self.n_decay_steps:
            lr = (
                0.5
                * math.cos(
                    (current_step - self.n_linear_steps)
                    / (self.n_decay_steps)
                    * math.pi
                )
                + 0.5
            ) * (self.lr_max - self.lr_final) + self.lr_final
        else:
            lr = self.lr_final
        return [lr for _ in self.base_lrs]


def freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        train_dataset = WaveformWithFramewiseLabelsDataset(
            wav_dir=cfg.train_wav_dir,
            teacher_dir=cfg.train_teacher_dir,
            wav_pattern=cfg.train_wav_pattern,
            teacher_pattern=cfg.train_teacher_pattern,
            max_duration=cfg.max_duration,
            normalize=cfg.normalize,
        )
        valid_dataset = WaveformWithFramewiseLabelsDataset(
            wav_dir=cfg.valid_wav_dir,
            teacher_dir=cfg.valid_teacher_dir,
            wav_pattern=cfg.valid_wav_pattern,
            teacher_pattern=cfg.valid_teacher_pattern,
            max_duration=cfg.max_duration,
            normalize=cfg.normalize,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dataset.collate,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=valid_dataset.collate,
        )

        self.model = self.model_setup()
        self.optimizer, self.scheduler = self.optimizer_setup()
        self.scaler = self.scaler_setup()

        self.run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            name=cfg.name,
            config=cfg.__dict__,
        )

        self.current_step = 0
        self.current_global_step = 0
        self.current_epoch = 0
        self.best_loss = math.inf
        self.pbar = None

    def model_setup(self) -> WavLMWithPredictionHead:
        model: WavLMWithPredictionHead = (
            WavLMWithPredictionHead.warmstart_from_pretrained_checkpoint(
                train_cfg=self.cfg, device=self.cfg.device
            )
        )

        # ===============================================================
        # YOU WILL NEED TO MODIFY THIS PART TO ENSURE THE RIGHT BEHAVIOUR
        # ===============================================================
        # * freeze feature extractor
        model.wavlm.feature_grad_mult = 0
        freeze(model.wavlm.feature_extractor)
        freeze(model.wavlm.layer_norm)
        # ---------------------------------------------------------------
        # * freeze feature extractor projection
        freeze(model.wavlm.post_extract_proj)
        # ---------------------------------------------------------------
        # * freeze mask embedding
        model.wavlm.mask_emb.requires_grad = False
        # ---------------------------------------------------------------
        # * freeze positional embeddings
        freeze(model.wavlm.encoder.pos_conv)
        # ---------------------------------------------------------------
        # * freeze the initial layernorm if not using prenorm
        if not model.wavlm.encoder.layer_norm_first:
            freeze(model.wavlm.encoder.layer_norm)
        # ---------------------------------------------------------------
        # * freeze the first few transformer layers
        NUM_TRANSFORMER_LAYERS_TO_FREEZE = 11
        for layer in model.wavlm.encoder.layers[:NUM_TRANSFORMER_LAYERS_TO_FREEZE]:
            freeze(layer)
        # ===============================================================

        return model

    def optimizer_setup(self) -> Tuple[Optimizer, LRScheduler]:
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = LinearRampCosineDecayScheduler(
            optimizer=optimizer,
            n_linear_steps=self.cfg.n_linear_steps,
            n_decay_steps=self.cfg.n_decay_steps,
            lr_init=self.cfg.lr_init,
            lr_max=self.cfg.lr_max,
            lr_final=self.cfg.lr_final,
        )
        return optimizer, scheduler

    def scaler_setup(self) -> Optional[torch.amp.GradScaler]:
        if self.cfg.dtype == torch.float16:
            return torch.amp.GradScaler(device=self.cfg.device)
        # don't use scaler for float or bfloat16
        return None

    def train_step(self, batch):
        wavs, labels, seqlens = batch
        wavs = wavs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        seqlens = seqlens.to(self.cfg.device)
        padding_mask = (
            torch.arange(labels.size(-1), device=self.cfg.device)[None, :]
            >= seqlens[:, None]
        )
        logits, mask = self.model(wavs, padding_mask, mask=True)
        minlen = min(logits.size(-2), labels.size(-1))
        logits = logits[:, :minlen, :]
        labels = labels[:, :minlen]
        mask = mask[:, :minlen]
        loss = torch.nn.functional.cross_entropy(logits[mask], labels[mask])
        return loss

    def valid_step(self, batch):
        wavs, labels, seqlens = batch
        wavs = wavs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        seqlens = seqlens.to(self.cfg.device)
        padding_mask = (
            torch.arange(labels.size(-1), device=self.cfg.device)[None, :]
            >= seqlens[:, None]
        )
        logits, mask = self.model(wavs, padding_mask, mask=True)
        minlen = min(logits.size(-2), labels.size(-1))
        logits = logits[:, :minlen, :]
        labels = labels[:, :minlen]
        mask = mask[:, :minlen]
        loss = torch.nn.functional.cross_entropy(logits[mask], labels[mask])
        accuracy = (logits[mask].argmax(-1) == labels[mask]).float().mean()
        return loss, accuracy

    def train_epoch(self) -> Generator[dict, None, None]:
        """Yields step information for one epoch of training"""
        self.model.train()
        self.optimizer.zero_grad()
        for loader_idx, batch in enumerate(self.train_loader):

            with amp.autocast(device_type=self.cfg.device, dtype=self.cfg.dtype):
                loss = self.train_step(batch)

                if self.cfg.accumulation_steps > 1:
                    loss / self.cfg.accumulation_steps

            if self.scaler is None:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()
            self.current_step += 1

            if (
                self.current_step + 1
            ) % self.cfg.accumulation_steps == 0 or loader_idx + 1 == len(
                self.train_loader
            ):
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip_max_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.current_global_step += 1
                self.pbar.update(1)
                yield loss.detach()
                self.model.train()

        if (loader_idx + 1) % self.cfg.accumulation_steps != 0:
            # in case there are leftover batches due to len(train_loader)
            # not being exactly divisible by self.accumulation steps
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.current_global_step += 1
            self.pbar.update(1)
            yield loss.detach()
            self.model.train()

        self.current_epoch += 1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses = []
        accuracies = []
        for batch in tqdm(
            self.valid_loader, desc="Validating", position=1, leave=False
        ):
            loss, accuracy = self.valid_step(batch)
            losses.append(loss.item())
            accuracies.append(accuracy.item())
        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        if loss <= self.best_loss:
            self.best_loss = loss
            self.save_models()
        else:
            print(f"val/loss {loss:.4f} was not in top 1")
        img_path = self.save_similarity_plot()
        return loss, accuracy, img_path

    def save_models(self):
        # save the full model with prediction head parameters
        checkpoint_path = Path(self.run.dir) / "best.full.pt"
        checkpoint = {
            "model": self.model.state_dict(),
            "wavlm_cfg": self.model.wavlm.cfg.__dict__,
            "train_cfg": self.cfg,
        }
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        # save only the wavlm model in the format of https://github.com/microsoft/unilm/tree/master/wavlm
        checkpoint_path = Path(self.run.dir) / "best.pt"
        checkpoint = {
            "model": self.model.wavlm.state_dict(),
            "cfg": self.model.wavlm.cfg.__dict__,
        }
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def save_similarity_plot(self):
        decoder = AudioDecoder("data/sample.flac", sample_rate=16000, num_channels=1)
        audio = decoder.get_all_samples()

        textgrid = tgt.read_textgrid("data/sample.TextGrid", include_empty_intervals=False)

        tMelSpectrogram = MelSpectrogram(16000, 1024, 400, 160, n_mels=100)
        tAmplitudeToDB = AmplitudeToDB(top_db=80)
        melspec = tAmplitudeToDB(tMelSpectrogram(audio.data))[0]

        with torch.inference_mode():
            features, _ = self.model.wavlm.extract_features(audio.data.to(self.cfg.device), output_layer=None)
        features = features.squeeze(0).cpu()

        features = torch.nn.functional.normalize(features, p=2, dim=1)
        similarity_matrix = features @ features.T
        similarity_matrix = similarity_matrix.cpu().numpy()

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 12),
            gridspec_kw={"height_ratios": [1, 8], "width_ratios": [1, 8]},
            constrained_layout=True,
        )

        # --- Top right ---
        ax1 = axes[0][1]
        ax1.imshow(melspec, aspect="auto", origin="lower")

        xticks = []
        xtickslabels = []
        for interval in textgrid.get_tier_by_name("syllables"):
            x1 = interval.start_time * 16000 / 160
            x2 = interval.end_time * 16000 / 160
            xticks.append((x1 + x2) / 2)
            xtickslabels.append(interval.text)
            ax1.axvline(x1, color="white")
            ax1.axvline(x2, color="white")
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xtickslabels, rotation=90)
        ax1.xaxis.set_ticks_position("top")
        ax1.xaxis.set_label_position("top")
        ax1.get_yaxis().set_visible(False)

        # --- Bottom right ---
        ax2 = axes[1][1]
        im = ax2.imshow(
            similarity_matrix, aspect="equal", origin="upper"
        )  # equal for square pixels
        ax2.axis("off")

        # --- top left ---
        axes[0][0].axis("off")

        # --- Bottom left ---
        ax3 = axes[1][0]
        ax3.imshow(melspec.T.flip(1), aspect="auto", origin="upper")
        yticks = []
        ytickslabels = []
        for interval in textgrid.get_tier_by_name("syllables"):
            y1 = interval.start_time * 16000 / 160
            y2 = interval.end_time * 16000 / 160
            yticks.append((y1 + y2) / 2)
            ytickslabels.append(interval.text)
            ax3.axhline(y1, color="white")
            ax3.axhline(y2, color="white")
        ax3.set_yticks(yticks)
        ax3.set_yticklabels(ytickslabels)
        ax3.get_xaxis().set_visible(False)

        img_path = Path(self.run.dir) / f"similarity-step-{self.current_global_step}.png"

        fig.savefig(img_path)

        return img_path


    def train(
        self,
        max_global_step: int,
        log_every_n_global_steps: int,
        validate_every_n_global_steps: int,
    ):
        """
        Training loop that runs until max epochs or steps is reached.
        Args:
            num_epochs: Maximum number of epochs to train for
            num_steps: Maximum number of steps to train for
            validate_every: Number of steps between validation runs
        """
        if self.current_global_step >= max_global_step:
            print(f"Already trained up to {self.current_global_step} steps")
            return
        
        self.model.eval()
        img_path = self.save_similarity_plot()
        self.run.log(data={"images/similarity": wandb.Image(img_path)}, step=self.current_global_step)
        self.model.train()

        self.pbar = tqdm(
            total=max_global_step, desc="Training", unit="step", position=0, leave=True
        )

        while True:
            for train_loss in self.train_epoch():
                log_data = {
                    "train/loss": train_loss,
                    "trainer/epoch": self.current_epoch,
                }

                log_flag = self.current_global_step % log_every_n_global_steps == 0

                if self.current_global_step % validate_every_n_global_steps == 0:
                    val_loss, val_accuracy, img_path = self.validate()
                    log_data["val/loss"] = val_loss
                    log_data["val/accuracy"] = val_accuracy
                    log_data["images/similarity"] = wandb.Image(img_path)
                    log_flag = True

                if log_flag:
                    self.run.log(data=log_data, step=self.current_global_step)
                    self.pbar.set_postfix(
                        {
                            "epoch": f"{self.current_epoch}",
                            "lr": f"{self.scheduler.get_lr()[0]:.1e}",
                            "train/loss": f"{log_data['train/loss']:.4f}",
                            "val/loss": f"{self.best_loss:.4f}",
                        }
                    )
                    self.pbar.update(0)

                if self.current_global_step >= max_global_step:
                    print(f"Reached max steps ({max_global_step})")
                    self.pbar.close()
                    return


if __name__ == "__main__":
    trainer = Trainer(TrainConfig())
    trainer.train(
        max_global_step=25000,
        log_every_n_global_steps=1,
        validate_every_n_global_steps=1000,
    )
