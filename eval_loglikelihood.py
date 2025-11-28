from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from zerosyl.ulm import ULM


class EncodedEvalDataset(Dataset):
    def __init__(self, segments_dir: str, tokenize_fn: Callable):
        self.segments_paths = sorted(list(Path(segments_dir).glob("*.pt")))
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
    batch: List[Tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[List[str], torch.Tensor, torch.Tensor, List[int]]:
    keys = [b[0] for b in batch]
    src_ids = torch.cat([b[1] for b in batch])
    tgt_ids = torch.cat([b[2] for b in batch])
    seqlens = [len(b[1]) for b in batch]
    return keys, src_ids, tgt_ids, seqlens


def compute_loglikelihoods(
    segments_dir: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
):
    model = ULM.from_pretrained_checkpoint(checkpoint_path).cuda()
    model.eval()

    print(f"Computing loglikelihoods for units in {segments_dir}...")

    dataset = EncodedEvalDataset(
        segments_dir=segments_dir,
        tokenize_fn=model.tokenize,
    )

    print(f"Found {len(dataset)} unit files")

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


if __name__ == "__main__":
    # activate zrc environment and run:
    # zrc submission:init sLM21 output/submissions/...

    compute_loglikelihoods(
        segments_dir="/home/nicolvisser/Workspace/zerosyl/output/segments/ZeroSylDiscrete-v040-k-10000/sLM21-dataset/lexical/dev",
        checkpoint_path="/home/nicolvisser/Workspace/zerosyl/wandb/offline-run-20251128_205820-5vcdyt5y/files/best.pt",
        output_path="output/submissions/test/lexical/dev.txt",
        batch_size=128,
        num_workers=23,
    )

    compute_loglikelihoods(
        segments_dir="/home/nicolvisser/Workspace/zerosyl/output/segments/ZeroSylDiscrete-v040-k-10000/sLM21-dataset/syntactic/dev",
        checkpoint_path="/home/nicolvisser/Workspace/zerosyl/wandb/offline-run-20251128_205820-5vcdyt5y/files/best.pt",
        output_path="output/submissions/test/syntactic/dev.txt",
        batch_size=128,
        num_workers=23,
    )

    # activate zrc environment and run:
    # zrc benchmarks:run sLM21 output/submissions/ulm-v0.4.0-units-10000 -s dev -t lexical syntactic
