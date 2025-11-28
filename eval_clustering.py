from pathlib import Path

import numpy as np
import tgt
import torch
from sklearn.metrics.cluster import contingency_matrix
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
    if not segments_dir.exists():
        print(f"{segments_dir} not found.")
        continue

    segments_paths = sorted(segments_dir.glob("dev*/**/*.pt"))
    assert len(segments_paths) > 0
    assert len(segments_paths) == 5567

    for sp, tp in zip(segments_paths, textgrid_paths):
        assert sp.stem == tp.stem

    labels_pred = []
    labels_true = []

    for sp, tp in zip(tqdm(segments_paths), textgrid_paths):
        segments = torch.load(sp).numpy()
        textgrid = tgt.read_textgrid(tp, include_empty_intervals=True)
        tier = textgrid.get_tier_by_name("syllables")

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

    # syllable purity
    sp = np.sum(p_y_given_z[y_star, np.arange(J)] * p_z)

    # cluster purity
    cp = np.sum(p_z_given_y[np.arange(I), z_star] * p_y)

    # syllable-normalized mutual information
    snmi = (
        -(p_yz * np.log(p_yz / p_y[:, None] / p_z[None, :])).sum()
        / (p_y * np.log(p_y)).sum()
    )

    print(system["name"])
    print(f"Syllable purity:                        {sp.item()*100:>10.1f}")
    print(f"Cluster purity:                         {cp.item()*100:>10.1f}")
    print(f"Syllable-normalized mutual information: {snmi.item()*100:>10.1f}")
    print()
