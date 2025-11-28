from pathlib import Path

import numpy as np
import tgt
import torch
from tqdm import tqdm

# fmt: off
systems = [
    {
        "name": "SylBoost 6.25",
        "segments_dir": "output/segments/syllablelm-625-official-ids-k-8192/LibriSpeech",
        "vocab_size": 8192
    },
    {
        "name": "Sylber",
        "segments_dir": "output/segments/sylber-custom-centroids-with-silence-k-10001/LibriSpeech",
        "vocab_size": 10001
    },
    {
        "name": "ZeroSyl w/o collapsing",
        "segments_dir": "output/segments/zerosyl-v040-with-silences-k-10000/LibriSpeech",
        "vocab_size": 10000
    },
    {
        "name": "ZeroSyl",
        "segments_dir": "output/segments/zerosyl-v040-collapsed-silences-k-9116/LibriSpeech",
        "vocab_size": 9116
    },
]
textgrid_dir = "data/alignments/LibriSpeech"
# fmt: on

textgrid_dir = Path(textgrid_dir)
textgrid_paths = sorted(textgrid_dir.glob("dev*/**/*.TextGrid"))
assert len(textgrid_paths) > 0
assert len(textgrid_paths) == 5567

for system in systems:

    segments_dir = Path(system["segments_dir"])
    segments_paths = sorted(segments_dir.glob("dev*/**/*.pt"))
    assert len(segments_paths) > 0
    assert len(segments_paths) == 5567

    for sp, tp in zip(segments_paths, textgrid_paths):
        assert sp.stem == tp.stem

    all_tokens = []
    all_durations = []

    for sp, tp in zip(tqdm(segments_paths), textgrid_paths):
        # process segments to find framewise tokens
        segments = torch.load(sp)
        starts, ends, tokens = segments.T.numpy()
        all_tokens.append(tokens)

        # process textgrid to find duration
        textgrid = tgt.read_textgrid(tp, include_empty_intervals=True)
        tier = textgrid.get_tier_by_name("syllables")
        duration = textgrid.end_time - textgrid.start_time
        all_durations.append(duration)

    all_tokens = np.concat(all_tokens, axis=0)
    total_duration = np.sum(all_durations)

    counts = np.bincount(all_tokens, minlength=system["vocab_size"])
    probs = counts / sum(counts) + 1e-10
    entropy = -np.sum(probs * np.log2(probs))
    bitrate = len(all_tokens) * entropy / total_duration

    bits_per_token = np.log2(8192)
    bitrate_naive = bits_per_token * len(all_tokens) / total_duration

    freq = len(all_tokens) / total_duration

    print(system["name"])
    print(f"Frequency [Hz]:  {freq:.2f}")
    print(f"Bitrate (entropic) [bps]:   {bitrate:.0f}")
    print(f"Bitrate (storage) [bps]:   {bitrate_naive:.0f}")
    print()
