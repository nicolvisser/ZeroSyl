from pathlib import Path

import pandas as pd


def eval_tsc(
    tsc_ikelihoods_path: str | Path,
):
    df = pd.read_csv(
        tsc_ikelihoods_path, delimiter=" ", header=None, names=["id", "loglikelihood"]
    )

    df = df.sort_values("id")

    def split_name(name: str):
        id, prefix_id, correct = name.split("_")
        if correct == "correct":
            correct = True
        elif correct == "incorrect":
            correct = False
        else:
            raise ValueError("Oops! found: " + correct)

        return id, prefix_id, correct

    df[["file_id", "prefix_id", "correct"]] = (
        df["id"].astype(str).apply(lambda s: pd.Series(split_name(s)))
    )

    scores = []

    for prefix_id in df["prefix_id"].unique():
        correct_ll = df[(df["prefix_id"] == prefix_id) & (df["correct"] == True)][
            "loglikelihood"
        ].item()
        incorrect_ll = df[(df["prefix_id"] == prefix_id) & (df["correct"] == False)][
            "loglikelihood"
        ].item()
        score = int(correct_ll > incorrect_ll)
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    return avg_score
