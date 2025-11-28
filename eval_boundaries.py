from pathlib import Path

import tgt
import torch
from tqdm import tqdm

from zerosyl.utils.boundaries import *

# fmt: off
TOLERANCE = 0.05  # seconds
systems = [
    {
        "name": "SylBoost 5.00Hz",
        "segments_dir": "output/segments/SylBoost500-k-16384/LibriSpeech",
        "constant_shift": -0.01,
    },
    {
        "name": "SylBoost 6.25Hz",
        "segments_dir": "output/segments/SylBoost625-k-8192/LibriSpeech",
        "constant_shift": -0.015,
    },
    {
        "name": "SylBoost 8.33Hz",
        "segments_dir": "output/segments/SylBoost833-k-2048",
        "constant_shift": -0.020,
    },
    {
        "name": "Sylber",
        "segments_dir": "output/segments/Sylber-k-10001/LibriSpeech",
        "constant_shift": -0.035,
    },
    {
        "name": "ZeroSyl Discrete",
        "segments_dir": "output/segments/ZeroSylDiscrete-v040-k-10000/LibriSpeech",
        "constant_shift": -0.005,
    },
    {
        "name": "ZeroSyl Collapse",
        "segments_dir": "output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech",
        "constant_shift": -0.005,
    },
]
textgrid_dir = "data/alignments/LibriSpeech"
# fmt: on


def split_utterance(
    seg: List[float], ref: tgt.IntervalTier, tolerance: float = TOLERANCE
) -> Tuple[List[List[float]], List[List[float]]]:

    ref_out = [
        list(ref_utt)
        for k, ref_utt in itertools.groupby(ref.intervals, lambda x: x.text != "")
        if k
    ]

    seg_out = []
    for ref_utt in ref_out:
        ref_utt_onset = ref_utt[0].start_time + tolerance
        ref_utt_offset = ref_utt[-1].end_time - tolerance
        seg_out.append([s for s in seg if ref_utt_onset < s < ref_utt_offset])
        seg_out[-1].append(ref_utt[-1].end_time)

    return seg_out, [
        [float(interval.end_time) for interval in ref_utt] for ref_utt in ref_out
    ]


textgrid_dir = Path(textgrid_dir)
textgrid_paths = sorted(textgrid_dir.glob("dev*/**/*.TextGrid"))
assert len(textgrid_paths) > 0
assert len(textgrid_paths) == 5567

for system in systems:
    segments_dir = Path(system["segments_dir"])
    if not segments_dir.exists():
        print(f"{segments_dir} not found.")
        continue

    segments_paths = sorted(segments_dir.glob("dev*/**/*.pt"))
    assert len(segments_paths) > 0
    assert len(segments_paths) == 5567

    for sp, tp in zip(segments_paths, textgrid_paths):
        assert sp.stem == tp.stem

    segs, refs = [], []

    for sp, tp in zip(tqdm(segments_paths), textgrid_paths):
        segments = torch.load(sp)
        textgrid = tgt.read_textgrid(tp, include_empty_intervals=True)

        starts, ends, _ = segments.T
        boundaries = starts.tolist() + ends.tolist()[-1:]
        boundaries = [round(b / 50.0 + system["constant_shift"], 2) for b in boundaries]

        seg, ref = split_utterance(
            boundaries,
            textgrid.get_tier_by_name("syllables"),
            tolerance=TOLERANCE,
        )
        segs.extend(seg)
        refs.extend(ref)

    # -------------- Calculate boundary evaluation metrics --------------

    precision, recall, f1 = eval_boundaries(segs, refs, tolerance=TOLERANCE)

    os = get_os(precision, recall)

    rvalue = get_rvalue(precision, recall)

    token_precision, token_recall, token_f1 = eval_token_boundaries(
        segs, refs, tolerance=TOLERANCE
    )

    print(system["name"])
    print(
        f"Precision: {precision*100:.0f}, Recall: {recall*100:.0f}, F1: {f1*100:.0f}, OS: {os*100:.0f}, R-value: {rvalue*100:.0f}"
    )
    print(
        f"Token Precision: {token_precision*100:.0f}, Token Recall: {token_recall*100:.0f}, Token F1: {token_f1*100:.0f}"
    )
    print()
