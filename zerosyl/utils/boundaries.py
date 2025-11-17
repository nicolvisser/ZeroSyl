import itertools
from typing import List, Tuple

import numpy as np


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
