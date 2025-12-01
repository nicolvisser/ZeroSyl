import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import wandb
from torch import amp, nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from zerosyl.ulm import ULM, ULMConfig


@dataclass
class TrainConfig:
    # --- Wandb details ---
    entity: str
    project: str
    name: str

    # --- General training control ---
    device: str
    dtype: torch.dtype
    accumulation_steps: int
    grad_clip_max_norm: float
    batch_size: int
    num_workers: int

    # --- Data configuration ---
    train_mmap_path: str
    valid_segments_dir: str
    valid_segments_pattern: str

    # --- ULM specific configuration ---
    train_ctx_win_size: int

    # --- Optimizer / learning rate schedule ---
    lr_init: float
    lr_max: float
    lr_final: float
    n_linear_steps: int
    n_decay_steps: int
    betas: tuple[float, float]
    weight_decay: float
    eps: float


class TokenizedUnitsChunkedMMapDataset(Dataset):
    def __init__(
        self,
        mmap_data_path: str,
        mmap_dtype: np.dtype,
        chunk_size: int,
        tokenize_fn: Callable,
    ):

        self.mmap = np.memmap(mmap_data_path, dtype=mmap_dtype, mode="r")
        assert self.mmap.ndim == 1
        self.tokenize_fn = tokenize_fn
        self.chunk_size = chunk_size

    def __len__(self):
        return self.mmap.shape[0] // self.chunk_size

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        max_start = len(self.mmap) - self.chunk_size
        start = np.random.randint(0, max_start)
        units = self.mmap[start : start + self.chunk_size]
        units = units.astype(np.long)
        units = torch.from_numpy(units)
        tokens = self.tokenize_fn(units)
        return tokens


class TokenizedUnitsUtteranceDataset(Dataset):
    def __init__(
        self,
        segments_dir: str,
        tokenize_fn: Callable,
        pattern: str = "**/*.pt",
    ):
        self.segments_paths = sorted(list(Path(segments_dir).glob(pattern)))
        assert len(self.segments_paths) > 0, "No segment files found"
        self.tokenize_fn = tokenize_fn

    def __len__(self) -> int:
        return len(self.segments_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        segments_path = self.segments_paths[idx]
        segments = torch.load(segments_path).long()
        units = segments[:, 2]
        tokens = self.tokenize_fn(units)
        return tokens


def collate_fn(
    tokens: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    src_tokens = [toks[:-1] for toks in tokens]
    tgt_tokens = [toks[1:] for toks in tokens]
    seqlens = [len(toks) for toks in src_tokens]
    src_tokens = torch.cat(src_tokens, dim=0)
    tgt_tokens = torch.cat(tgt_tokens, dim=0)
    assert sum(seqlens) == src_tokens.size(0)
    return src_tokens, tgt_tokens, seqlens


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


class Trainer:
    def __init__(self, model_cfg: ULMConfig, train_cfg: TrainConfig):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.model = ULM(model_cfg).to(self.train_cfg.device)

        # initialize optimizer with training config lr and params
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr_init,
            betas=self.train_cfg.betas,
            weight_decay=self.train_cfg.weight_decay,
            eps=self.train_cfg.eps,
        )

        self.scheduler = LinearRampCosineDecayScheduler(
            optimizer=self.optimizer,
            n_linear_steps=self.train_cfg.n_linear_steps,
            n_decay_steps=self.train_cfg.n_decay_steps,
            lr_init=self.train_cfg.lr_init,
            lr_max=self.train_cfg.lr_max,
            lr_final=self.train_cfg.lr_final,
        )

        # Use GradScaler only for float16; bfloat16 uses autocast without scaler
        self.scaler = (
            torch.amp.GradScaler() if self.train_cfg.dtype == torch.float16 else None
        )

        train_dataset = TokenizedUnitsChunkedMMapDataset(
            mmap_data_path=train_cfg.train_mmap_path,
            mmap_dtype=np.uint16,
            chunk_size=train_cfg.train_ctx_win_size,
            tokenize_fn=self.model.tokenize,
        )
        valid_dataset = TokenizedUnitsUtteranceDataset(
            segments_dir=train_cfg.valid_segments_dir,
            tokenize_fn=self.model.tokenize,
            pattern=train_cfg.valid_segments_pattern,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.run = wandb.init(
            entity=train_cfg.entity,
            project=train_cfg.project,
            name=train_cfg.name,
            config={
                "model_cfg": model_cfg.__dict__,
                "train_cfg": train_cfg.__dict__,
            },
        )

        self.current_step = 0
        self.current_global_step = 0
        self.current_epoch = 0
        self.best_loss = math.inf
        self.pbar = None

    def train_step(self, batch):
        src_tokens, tgt_tokens, seqlens = batch
        src_tokens = src_tokens.to(self.train_cfg.device)
        tgt_tokens = tgt_tokens.to(self.train_cfg.device)
        logits = self.model.forward(src_tokens, seqlens)
        loss = torch.nn.functional.cross_entropy(logits, tgt_tokens)
        return loss

    def valid_step(self, batch):
        src_tokens, tgt_tokens, seqlens = batch
        src_tokens = src_tokens.to(self.train_cfg.device)
        tgt_tokens = tgt_tokens.to(self.train_cfg.device)
        logits = self.model.forward(src_tokens, seqlens)
        loss = torch.nn.functional.cross_entropy(logits, tgt_tokens)
        accuracy = (logits.argmax(-1) == tgt_tokens).float().mean()
        return loss, accuracy

    def train_epoch(self):
        """Yields step information for one epoch of training"""
        self.model.train()
        self.optimizer.zero_grad()
        for loader_idx, batch in enumerate(self.train_loader):

            with amp.autocast(
                device_type=self.train_cfg.device, dtype=self.train_cfg.dtype
            ):
                loss = self.train_step(batch)

                if self.train_cfg.accumulation_steps > 1:
                    loss = loss / self.train_cfg.accumulation_steps

            if self.scaler is None:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()

            loss = loss * self.train_cfg.accumulation_steps

            self.current_step += 1

            # Step optimizer when we've accumulated enough gradients or at epoch end
            if (
                self.current_step % self.train_cfg.accumulation_steps == 0
                or loader_idx + 1 == len(self.train_loader)
            ):
                # unscale before clipping if using scaler
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.train_cfg.grad_clip_max_norm
                )

                if self.scaler is None:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

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

        if loss >= self.best_loss:
            print("Validation loss did not improve.")
        else:
            self.best_loss = loss

            checkpoint_path = Path(self.run.dir) / f"best.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "cfg": self.model.cfg.__dict__,
            }
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)

        return loss, accuracy

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

        self.pbar = tqdm(
            total=max_global_step, desc="Training", unit="step", position=0, leave=True
        )

        while True:
            for train_loss in self.train_epoch():
                log_data = {
                    "train/loss": train_loss,
                    "trainer/epoch": self.current_epoch,
                    "trainer/lr": self.scheduler.get_lr()[0],
                }

                log_flag = self.current_global_step % log_every_n_global_steps == 0

                if self.current_global_step % validate_every_n_global_steps == 0:
                    val_loss, val_accuracy = self.validate()
                    log_data["val/loss"] = val_loss
                    log_data["val/accuracy"] = val_accuracy
                    log_flag = True

                if log_flag:
                    self.run.log(data=log_data, step=self.current_global_step)
                    self.pbar.set_postfix(
                        {
                            "epoch": f"{self.current_epoch}",
                            "lr": f"{self.scheduler.get_lr()[0]:.1e}",
                            "train/loss": f"{train_loss:.4f}",
                            "val/loss": f"{self.best_loss:.4f}",
                        }
                    )
                    self.pbar.update(0)

                if self.current_global_step >= max_global_step:
                    print(f"Reached max steps ({max_global_step})")
                    checkpoint_path = (
                        Path(self.run.dir) / f"step-{self.current_global_step}.pt"
                    )
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "cfg": self.model.cfg.__dict__,
                    }
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)
                    self.pbar.close()
                    self.run.finish()
                    return


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model_cfg = ULMConfig(
        vocab_size=9116,
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=4 * 768,
        n_heads=12,
        n_kv_heads=12,
        dropout=0.1,
        norm_eps=1e-6,
        rope_theta=10000.0,
    )

    train_cfg = TrainConfig(
        entity="zerospeech",
        project="zerosyl-ulm",
        name=f"ULM-LL-ZeroSylCollapsed-v040-k-9116",
        device="cuda",
        dtype=torch.bfloat16,
        accumulation_steps=4,
        grad_clip_max_norm=1.0,
        batch_size=10,
        num_workers=23,
        train_mmap_path="/mnt/newt/zerosyl/v0.4.0/librilight-zerosyl-v040-tokens-k-9116.bin",
        valid_segments_dir="/home/nicolvisser/Workspace/zerosyl/output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech",
        valid_segments_pattern="dev*/**/*.pt",
        train_ctx_win_size=2048,
        lr_init=0.0,
        lr_max=2e-4,
        lr_final=2e-5,
        n_linear_steps=16_000,
        n_decay_steps=200_000 - 16_000,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        eps=1e-8,
    )

    trainer = Trainer(model_cfg, train_cfg)

    trainer.train(
        max_global_step=200_000,
        log_every_n_global_steps=1,
        validate_every_n_global_steps=1_000,
    )
