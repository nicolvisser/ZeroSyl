from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from zerosyl.ulm import ULM


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


class EncodedEvalDataset(Dataset):
    def __init__(
        self, segments_dir: str, tokenize_fn: Callable, segments_pattern: str = "*.pt"
    ):
        self.segments_paths = sorted(list(Path(segments_dir).glob(segments_pattern)))
        self.tokenize_fn = tokenize_fn

    def __len__(self):
        return len(self.segments_paths)

    def __getitem__(self, idx: int):
        segments_path = self.segments_paths[idx]
        key = segments_path.stem
        segments = torch.load(segments_path).long()
        _, _, units = segments.T
        tokens = self.tokenize_fn(units)
        src_tokens = tokens[:-1].clone()
        tgt_tokens = tokens[1:].clone()
        return key, src_tokens, tgt_tokens


def collate_fn(
    batch: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, list[int]]:
    keys = [b[0] for b in batch]
    src_ids = torch.cat([b[1] for b in batch])
    tgt_ids = torch.cat([b[2] for b in batch])
    seqlens = [len(b[1]) for b in batch]
    return keys, src_ids, tgt_ids, seqlens


def compute_loglikelihoods(
    segments_dir: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    segments_pattern: str = "*.pt",
):
    if checkpoint_path is None:
        model = ULM.from_remote().cuda()
    elif is_url(str(checkpoint_path)):
        model = ULM.from_remote(str(checkpoint_path)).cuda()
    else:
        model = ULM.from_pretrained_checkpoint(checkpoint_path).cuda()

    model.eval()
    num_params = sum(map(torch.numel, model.parameters()))
    print(f"Model loaded with {num_params,} parameters.")

    print(f"Computing loglikelihoods for units in {segments_dir}...")

    dataset = EncodedEvalDataset(
        segments_dir=segments_dir,
        tokenize_fn=model.tokenize,
        segments_pattern=segments_pattern,
    )

    print(f"Found {len(dataset)} segment files with pattern {segments_pattern}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    data = []
    with torch.inference_mode():
        for keys, src_ids, tgt_ids, seqlens in tqdm(dataloader):
            logits = model.forward(src_ids.cuda(), seqlens=seqlens)
            neg_obs_log_probs = torch.nn.functional.cross_entropy(
                logits, tgt_ids.cuda(), reduction="none"
            )

            starts = [sum(seqlens[:i]) for i in range(len(seqlens))]
            stops = [starts[i] + seqlens[i] for i in range(len(seqlens))]

            for key, start, stop in zip(keys, starts, stops):
                ll = -neg_obs_log_probs[start:stop].sum().item()
                data.append((key, ll))

    strings = [f"{key} {ll}" for key, ll in data]
    string = "\n".join(strings)
    Path(output_path).write_text(string)
