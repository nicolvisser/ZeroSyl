from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rapidfuzz.distance import Levenshtein
from torchcodec.decoders import AudioDecoder

transcribed_root = Path("output/transcribed")
transcribed_gt_dir = Path("output/transcribed/original-waveforms")

stems_4to10_seconds = set()

waveform_dir = Path(
    "/home/nicolvisser/Workspace/zerosyl/data/waveforms/LibriSpeech/dev-clean"
)
for waveform_path in waveform_dir.glob("**/*.flac"):
    duration = AudioDecoder(waveform_path).metadata.duration_seconds_from_header
    if duration >= 4 and duration <= 10:
        stems_4to10_seconds.add(waveform_path.stem)


def compute_wer(hyp, ref):
    """
    hyp: hypothesis transcription (model output)
    ref: ground-truth transcription
    """
    # Handle missing data
    if not isinstance(hyp, str) or not isinstance(ref, str):
        return None

    hyp_words = hyp.strip().split()
    ref_words = ref.strip().split()

    # Levenshtein distance on word sequences
    dist = Levenshtein.distance(hyp_words, ref_words)
    return dist / max(1, len(ref_words))


def compute_cer(hyp, ref):
    """
    hyp: hypothesis transcription (model output)
    ref: ground-truth transcription
    """
    if not isinstance(hyp, str) or not isinstance(ref, str):
        return None

    hyp_chars = list(hyp.strip())
    ref_chars = list(ref.strip())

    dist = Levenshtein.distance(hyp_chars, ref_chars)
    return dist / max(1, len(ref_chars))


def get_encoder_id(ulm_id: str) -> str:
    return ulm_id.split("ELLA-V-")[-1].split("-neucodec")[0]


def get_train_data_amount(ulm_id: str) -> int:
    return f"{ulm_id.split("-")[-1]} h"


def simplify_encoder_id(name):
    if name.startswith("Sylber"):
        return "Sylber\n[53 bps]"
    elif name.startswith("SylBoost"):
        return "SylBoost\n[73 bps]"
    elif name.startswith("ZeroSyl"):
        return "ZeroSyl\n[52 bps]"
    else:
        return name


gt_data = []
for transcription_path in sorted(transcribed_gt_dir.glob("LibriSpeech/dev*/**/*.txt")):
    gt_data.append(
        {
            "filename": transcription_path.stem,
            "transcription": transcription_path.read_text(),
        }
    )
df_gt = pd.DataFrame(gt_data)


transcribed_data = []
for dir in transcribed_root.iterdir():
    # skip non-directories and skip the GT directory
    if not dir.is_dir() or dir == transcribed_gt_dir:
        continue

    for transcription_path in sorted(dir.glob("LibriSpeech/dev*/**/*.txt")):
        transcribed_data.append(
            {
                "ulm-model-id": dir.stem,
                "filename": transcription_path.stem,
                "transcription": transcription_path.read_text(),
            }
        )
df_transcribed = pd.DataFrame(transcribed_data)

df_merged = df_transcribed.merge(
    df_gt.rename(columns={"transcription": "gt_transcription"}),
    on="filename",
    how="outer",
)


df_merged["wer"] = df_merged.apply(
    lambda row: compute_wer(row["transcription"], row["gt_transcription"]), axis=1
)
df_merged["wer"] = df_merged["wer"] * 100

df_merged["cer"] = df_merged.apply(
    lambda row: compute_cer(row["transcription"], row["gt_transcription"]), axis=1
)
df_merged["cer"] = df_merged["cer"] * 100

df_merged["encoder_id"] = df_merged["ulm-model-id"].apply(get_encoder_id)
df_merged["train_data_amount"] = df_merged["ulm-model-id"].apply(get_train_data_amount)


df_merged.sort_values(by=["wer"], ascending=False).head()

df_merged["encoder_short"] = df_merged["encoder_id"].map(simplify_encoder_id)


# REMOVE OUTLIERS

# --- WER ---
Q1_wer = df_merged["wer"].quantile(0.25)
Q3_wer = df_merged["wer"].quantile(0.75)
IQR_wer = Q3_wer - Q1_wer
upper_wer = Q3_wer + 1.5 * IQR_wer

# --- CER ---
Q1_cer = df_merged["cer"].quantile(0.25)
Q3_cer = df_merged["cer"].quantile(0.75)
IQR_cer = Q3_cer - Q1_cer
upper_cer = Q3_cer + 1.5 * IQR_cer

# Keep rows that are within BOTH ranges
df_clean = df_merged[
    (df_merged["wer"] <= upper_wer) & (df_merged["cer"] <= upper_cer)
].copy()

sns.set_theme(style="dark", font="monospace")


# df_plot1 = df_clean.melt(
#     id_vars=["encoder_id", "ulm-model-id", "filename", "train_data_amount", "encoder_short"],
#     value_vars=["wer", "cer"],
#     var_name="error_type",
#     value_name="error_rate"
# )
# df_plot1["error_type"] = df_plot1["error_type"].str.upper()
# df_plot1 = df_plot1[df_plot1["train_data_amount"] == "460 h"]
# df_plot1 = df_plot1.sort_values("encoder_short")

# sns.violinplot(
#     data=df_plot1,
#     x="encoder_short",
#     y="error_rate",
#     hue="error_type",
#     split=True,
#     inner="quart",
#     fill=True,
#     cut=0,
# )

# #plt.xticks(rotation=45, ha="right")
# plt.xlabel("Tokenization method")
# plt.legend(title="Evaluation type")
# plt.ylabel("Error rate (%)")
# plt.tight_layout()
# plt.savefig("option1.png")
# plt.show()


df_plot2 = df_clean.copy()
df_plot2 = df_plot2[df_plot2["train_data_amount"].isin(["100 h", "460 h"])]
df_plot2 = df_plot2.sort_values(["encoder_short", "train_data_amount"])

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(
    figsize=(6, 3), constrained_layout=True
)  # Widened the figure for the horizontal legend
ax = plt.subplot()
sns.violinplot(
    data=df_plot2,
    x="encoder_short",
    y="wer",
    hue="train_data_amount",
    split=True,
    inner="quart",
    fill=True,
    bw_adjust=0.75,
    cut=0,
    gridsize=1000,
    ax=ax,
)
ax.set_xlabel(None)
ax.set_ylabel("Word error rate (%)", fontweight="bold")

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

# make a legend that is one line: title + *entries
handles, labels = ax.get_legend_handles_labels()
empty_handle = plt.plot([], marker="", ls="")[0]
handles = [empty_handle] + handles
labels = ["Amount of training data:"] + labels

legend = ax.legend(
    handles=handles,
    labels=labels,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.46, 1.15),
    frameon=False,
)
for text in legend.get_texts():
    text.set_fontweight("bold")

plt.savefig("intelligibility.pdf", bbox_inches="tight")


df_4to10 = df_merged[df_merged["filename"].isin(stems_4to10_seconds)]


mean_wer_df = df_4to10.groupby(["encoder_id", "train_data_amount"], as_index=False)[
    "wer"
].mean()

print(mean_wer_df)

mean_cer_df = df_4to10.groupby(["encoder_id", "train_data_amount"], as_index=False)[
    "cer"
].mean()

print(mean_cer_df)
