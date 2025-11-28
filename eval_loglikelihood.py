from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from zerosyl.ulm import ULM


class EncodedEvalDataset(Dataset):
    def __init__(self, data_dir: str, tokenize_fn: Callable, dedupe: bool = False):
        self.units_paths = sorted(list(Path(data_dir).glob("*.pt")))
        self.tokenize_fn = tokenize_fn
        self.dedupe = dedupe

    def __len__(self):
        return len(self.units_paths)

    def __getitem__(self, idx: int):
        units_path = self.units_paths[idx]
        key = units_path.stem
        units = torch.load(units_path).long()
        tokens = self.tokenize_fn(units)
        if self.dedupe:
            tokens = torch.unique_consecutive(tokens)
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
    units_dir: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    dedupe: bool = True,
):
    model = ULM.from_pretrained_checkpoint(checkpoint_path).cuda()
    model.eval()

    print(f"Computing loglikelihoods for units in {units_dir}...")
    print(f"Dedupe? {dedupe}")

    dataset = EncodedEvalDataset(
        data_dir=units_dir,
        tokenize_fn=model.tokenize,
        dedupe=dedupe,
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
    # zrc submission:init sLM21 output/submissions/ulm-v0.4.0-units-10000/

    compute_loglikelihoods(
        units_dir="/home/nicolvisser/Data/zerosyl/v0.4.0/ulm-units-10000/sLM21-dataset/lexical/dev",
        checkpoint_path="/home/nicolvisser/Workspace/zerosyl/wandb/run-20251118_202437-2hhkxoxk/files/step-3000.pt",
        output_path="output/submissions/ulm-v0.4.0-units-10000/lexical/dev.txt",
        batch_size=128,
        num_workers=23,
        dedupe=False,
    )

    compute_loglikelihoods(
        units_dir="/home/nicolvisser/Data/zerosyl/v0.4.0/ulm-units-10000/sLM21-dataset/syntactic/dev",
        checkpoint_path="/home/nicolvisser/Workspace/zerosyl/wandb/run-20251118_202437-2hhkxoxk/files/step-3000.pt",
        output_path="output/submissions/ulm-v0.4.0-units-10000/syntactic/dev.txt",
        batch_size=128,
        num_workers=23,
        dedupe=False,
    )

    # activate zrc environment and run:
    # zrc benchmarks:run sLM21 output/submissions/ulm-v0.4.0-units-10000 -s dev -t lexical syntactic
