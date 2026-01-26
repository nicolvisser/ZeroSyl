import itertools
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tgt
import torch
from rich.progress import track


def evaluate_boundary_metrics(
    segments_dir: str | Path,
    textgrid_dir: str | Path,
    segments_pattern: str = "dev*/**/*.pt",
    textgrid_pattern: str = "dev*/**/*.TextGrid",
    frame_rate: float = 50.0,
    constant_shift: float = 0.0,
    tolerance: float = 0.05,
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

    segs, refs = [], []

    for sp, tp in track(
        zip(segments_paths, textgrid_paths),
        description="Calculating...",
        total=len(segments_paths),
    ):
        segments = torch.load(sp)
        textgrid = tgt.read_textgrid(tp, include_empty_intervals=True)

        starts, ends, _ = segments.T
        boundaries = starts.tolist() + ends.tolist()[-1:]
        boundaries = [round(b / frame_rate + constant_shift, 2) for b in boundaries]

        seg, ref = split_utterance(
            boundaries,
            textgrid.get_tier_by_name("syllables"),
            tolerance=tolerance,
        )
        segs.extend(seg)
        refs.extend(ref)

    # -------------- Calculate boundary evaluation metrics --------------

    precision, recall, f1 = eval_boundaries(segs, refs, tolerance=tolerance)

    os = get_os(precision, recall)

    rvalue = get_rvalue(precision, recall)

    token_precision, token_recall, token_f1 = eval_token_boundaries(
        segs, refs, tolerance=tolerance
    )

    return (precision, recall, f1, os, rvalue, token_precision, token_recall, token_f1)


def split_utterance(
    seg: List[float], ref: tgt.IntervalTier, tolerance: float
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


def get_p_r_f1(n_seg: int, n_ref: int, n_hit: int) -> Tuple[float, float, float]:

    if n_seg == n_ref == 0:
        return 0, 0, -np.inf
    elif n_hit == 0:
        return 0, 0, 0

    if n_seg != 0:
        precision = float(n_hit / n_seg)
    else:
        precision = np.inf

    if n_ref != 0:
        recall = float(n_hit / n_ref)
    else:
        recall = np.inf

    if precision + recall != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = -np.inf

    return precision, recall, f1_score


def eval_boundaries(
    seg: List[float], ref: List[float], tolerance: float = 0.05
) -> Tuple[float, float, float]:

    n_seg, n_ref, n_hit = 0, 0, 0
    assert len(seg) == len(ref)

    for i_utterance in range(len(seg)):
        prediction = seg[i_utterance]
        ground_truth = ref[i_utterance]

        if (
            len(prediction) > 0
            and len(ground_truth) > 0
            and abs(prediction[-1] - ground_truth[-1]) <= tolerance
        ):
            prediction = prediction[:-1]
            if len(ground_truth) > 0:
                ground_truth = ground_truth[:-1]

        n_seg += len(prediction)
        n_ref += len(ground_truth)

        if len(prediction) == 0 or len(ground_truth) == 0:
            continue

        for i_ref in ground_truth:
            for i, i_seg in enumerate(prediction):
                if abs(i_ref - i_seg) <= tolerance:
                    n_hit += 1
                    prediction.pop(i)
                    break

    return get_p_r_f1(n_seg, n_ref, n_hit)


def get_os(precision: float, recall: float) -> float:

    if precision == 0:
        return -np.inf
    else:
        return recall / precision - 1


def get_rvalue(precision: float, recall: float) -> float:

    os = get_os(precision, recall)
    r1 = np.sqrt((1 - recall) ** 2 + os**2)
    r2 = (-os + recall - 1) / np.sqrt(2)

    return 1 - (np.abs(r1) + np.abs(r2)) / 2


def eval_token_boundaries(
    seg: List[float], ref: List[float], tolerance: float = 0.05
) -> Tuple[float, float, float]:

    n_tokens_seg, n_tokens_ref, n_tokens_hit = 0, 0, 0
    assert len(seg) == len(ref)

    for i_utterance in range(len(seg)):
        prediction = seg[i_utterance]
        ground_truth = ref[i_utterance]

        seg_segments = [(a, b) for a, b in itertools.pairwise([0] + prediction)]
        ref_segments = [(a, b) for a, b in itertools.pairwise([0] + ground_truth)]

        ref_intervals = []
        for word_start, word_end in ref_segments:
            ref_intervals.append(
                (
                    (max(0, word_start - tolerance), word_start + tolerance),
                    (word_end - tolerance, word_end + tolerance),
                )
            )

        n_tokens_ref += len(ref_intervals)
        n_tokens_seg += len(seg_segments)

        for seg_start, seg_end in seg_segments:
            for i_segment, (ref_start_interval, ref_end_interval) in enumerate(
                ref_intervals
            ):
                ref_start_lower, ref_start_upper = ref_start_interval
                ref_end_lower, ref_end_upper = ref_end_interval

                if (
                    ref_start_lower <= seg_start <= ref_start_upper
                    and ref_end_lower <= seg_end <= ref_end_upper
                ):
                    n_tokens_hit += 1
                    ref_intervals.pop(i_segment)
                    break

    return get_p_r_f1(n_tokens_seg, n_tokens_ref, n_tokens_hit)
