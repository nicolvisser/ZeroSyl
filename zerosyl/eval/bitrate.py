from pathlib import Path

import numpy as np
import tgt
import torch
from rich.progress import track


def evaluate_bitrate_and_freq(
    segments_dir: str | Path,
    textgrid_dir: str | Path,
    segments_pattern: str = "dev*/**/*.pt",
    textgrid_pattern: str = "dev*/**/*.TextGrid",
    vocab_size: int | None = None,
):
    segments_dir = Path(segments_dir)
    textgrid_dir = Path(textgrid_dir)

    segments_paths = sorted(segments_dir.glob(segments_pattern))
    textgrid_paths = sorted(textgrid_dir.glob(textgrid_pattern))

    assert len(segments_paths) > 0
    assert len(textgrid_paths) > 0
    assert len(segments_paths) == len(textgrid_paths)
    for sp, tp in zip(segments_paths, textgrid_paths):
        assert sp.stem == tp.stem

    all_ids = []
    all_durations = []

    for sp, tp in track(
        zip(segments_paths, textgrid_paths),
        description="Calculating...",
        total=len(segments_paths),
    ):
        # fetch ids from segment file
        segments = torch.load(sp)
        starts, ends, ids = segments.T.numpy()
        all_ids.append(ids)

        # fetch duration from textgrid
        textgrid = tgt.read_textgrid(tp, include_empty_intervals=True)
        duration = textgrid.end_time - textgrid.start_time
        all_durations.append(duration)

    all_ids = np.concat(all_ids, axis=0)
    total_duration = np.sum(all_durations)

    if vocab_size is None:
        vocab_size = all_ids.max() + 1
    counts = np.bincount(all_ids, minlength=vocab_size)
    probs = counts / sum(counts) + 1e-10
    entropy = -np.sum(probs * np.log2(probs))
    bitrate = len(all_ids) * entropy / total_duration

    freq = len(all_ids) / total_duration

    return bitrate, freq
