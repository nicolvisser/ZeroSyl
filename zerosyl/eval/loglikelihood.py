from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import OPTConfig, OPTForCausalLM


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


class EncodedEvalDataset(Dataset):
    def __init__(
        self,
        segments_dir: str,
        tokenize_fn: Callable,
        pad_id: int,
        segments_pattern: str = "*.pt",
    ):
        self.segments_paths = sorted(list(Path(segments_dir).glob(segments_pattern)))
        self.tokenize_fn = tokenize_fn
        self.PAD_ID = pad_id

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
        self,
        batch: list[tuple[str, torch.Tensor, torch.Tensor]],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, list[int]]:
        keys = [b[0] for b in batch]
        src_ids = pad_sequence(
            [b[1] for b in batch], batch_first=True, padding_value=self.PAD_ID
        )
        tgt_ids = pad_sequence(
            [b[2] for b in batch], batch_first=True, padding_value=self.PAD_ID
        )
        seqlens = [len(b[1]) for b in batch]
        return keys, src_ids, tgt_ids, seqlens


def compute_loglikelihoods(
    segments_dir: str | Path,
    output_path: str | Path,
    checkpoint_path: (
        str | Path
    ) = "https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-60kh-ZeroSylCollapsed-v040-k-9116.pt",
    batch_size: int = 64,
    num_workers: int = 8,
    segments_pattern: str = "*.pt",
    normalize=False,
):
    if is_url(str(checkpoint_path)):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path)

    cfg = OPTConfig(**checkpoint["cfg"])
    model = OPTForCausalLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to("cuda")
    num_params = sum(map(torch.numel, model.parameters()))
    print(f"Model loaded with {num_params:,} parameters.")

    def tokenize(units: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat(
            [
                torch.tensor(
                    [model.config.bos_token_id], dtype=torch.long, device=units.device
                ),
                units.long(),
            ]
        )
        return tokens

    print(f"Computing loglikelihoods for units in {segments_dir}...")
    dataset = EncodedEvalDataset(
        segments_dir=segments_dir,
        tokenize_fn=tokenize,
        pad_id=model.config.pad_token_id,
        segments_pattern=segments_pattern,
    )

    print(f"Found {len(dataset)} segment files with pattern {segments_pattern}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
    )

    data = []
    with torch.inference_mode():
        for keys, src_ids, tgt_ids, seqlens in tqdm(dataloader):
            bsz = src_ids.size(0)
            logits = model.forward(src_ids.cuda()).logits
            neg_obs_log_probs = torch.nn.functional.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=tgt_ids.view(-1).cuda(),
                reduction="none",
            ).view(bsz, -1)

            for b, (key, seqlen) in enumerate(zip(keys, seqlens)):
                ll = -neg_obs_log_probs[b, :seqlen].sum().item()
                if normalize:
                    ll /= seqlen
                data.append((key, ll))

    strings = [f"{key} {ll}" for key, ll in data]
    string = "\n".join(strings)
    Path(output_path).write_text(string)
