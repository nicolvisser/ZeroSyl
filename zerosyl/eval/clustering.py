from pathlib import Path

import numpy as np
import tgt
import torch
from rich.progress import track
from sklearn.metrics.cluster import contingency_matrix


def evaluate_clustering_metrics(
    segments_dir: str | Path,
    textgrid_dir: str | Path,
    segments_pattern: str = "dev*/**/*.pt",
    textgrid_pattern: str = "dev*/**/*.TextGrid",
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

    labels_pred = []
    labels_true = []

    for sp, tp in track(
        zip(segments_paths, textgrid_paths),
        description="Calculating...",
        total=len(segments_paths),
    ):
        segments = torch.load(sp).numpy()
        textgrid = tgt.read_textgrid(tp, include_empty_intervals=True)
        tier: tgt.Tier = textgrid.get_tier_by_name("syllables")

        for segment in segments:
            s_start = segment[0] / 50
            s_end = segment[1] / 50
            intervals = tier.get_annotations_between_timepoints(
                start=s_start, end=s_end, left_overlap=True, right_overlap=True
            )

            max_overlap_duration = -1
            max_overlap_label = None
            for i in intervals:
                i_start = max(i.start_time, s_start)
                i_end = min(i.end_time, s_end)
                if i_end - i_start > max_overlap_duration:
                    max_overlap_duration = i_end - i_start
                    max_overlap_label = i.text

            label_pred = segment[2].item()
            label_true = max_overlap_label

            labels_pred.append(label_pred)
            labels_true.append(label_true)

    labels_pred = np.array(labels_pred)
    labels_true = np.array(labels_true)

    joint_counts = contingency_matrix(labels_true, labels_pred)

    # joint probabilities
    p_yz = joint_counts / joint_counts.sum() + 1e-10  # (I,J)
    I, J = p_yz.shape

    # marginal probabilities
    p_y = np.sum(p_yz, axis=1)  # (I,)
    p_z = np.sum(p_yz, axis=0)  # (J,)

    # most likely target label
    z_star = np.argmax(p_yz, axis=1)  # (I,)
    # most likely syllable label
    y_star = np.argmax(p_yz, axis=0)  # (J,)

    # conditional probabilities
    p_y_given_z = p_yz / p_z[None, :]  # (I,J)
    p_z_given_y = p_yz / p_y[:, None]  # (I,J)

    # per-cluster purity, a.k.a. normal purity a.k.a syllable purity
    per_cluster_purity = np.sum(p_y_given_z[y_star, np.arange(J)] * p_z)

    # per-syllable purity, a.k.a. reverse purity a.k.a cluster purity
    per_syllable_purity = np.sum(p_z_given_y[np.arange(I), z_star] * p_y)

    # syllable-normalized mutual information
    snmi = (
        -(p_yz * np.log(p_yz / p_y[:, None] / p_z[None, :])).sum()
        / (p_y * np.log(p_y)).sum()
    )

    return per_cluster_purity, per_syllable_purity, snmi
