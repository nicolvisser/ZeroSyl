from pathlib import Path

import numpy as np
import tgt
import torch
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from zerosyl.model import ZeroSylDiscrete

checkpoint_path = Path("checkpoints/WavLM-Large.pt")
centroids_path = Path("checkpoints/km10000-centroids-v030.pt")
waveform_dir = Path("data/waveforms/LibriSpeech")
alignment_dir = Path("data/alignments/LibriSpeech")

waveform_paths = sorted(waveform_dir.glob("dev*/**/*.flac"))
alignment_paths = sorted(alignment_dir.glob("dev*/**/*.TextGrid"))

assert len(waveform_paths) == len(alignment_paths) == 5567
for wp, ap in zip(waveform_paths, alignment_paths):
    assert wp.stem == ap.stem

model = ZeroSylDiscrete.from_pretrained_checkpoint(
    checkpoint_path, centroids_path
).cuda()

all_syllables_framewise = []
all_tokens_framewise = []
all_tokens_deduped = []

for waveform_path, alignment_path in zip(
    tqdm(waveform_paths, desc="loading data"), alignment_paths
):
    # extract ground truth syllables from the alignment file
    tg = tgt.read_textgrid(alignment_path, include_empty_intervals=True)
    tier = tg.get_tier_by_name("syllables")
    timesteps = np.arange(0.5 / 100, tg.end_time, 1 / 100)  # 100 Hz
    syllables_framewise = [tier.get_annotations_by_time(t)[0].text for t in timesteps]
    syllables_framewise = [("SIL" if s == "" else s) for s in syllables_framewise]
    syllables_framewise = np.array(syllables_framewise)

    # extract tokens from the speech
    decoder = AudioDecoder(waveform_path, sample_rate=16000, num_channels=1)
    audio = decoder.get_all_samples()
    tokens, starts, ends = model.tokenize(audio.data.cuda())
    tokens_framewise = torch.repeat_interleave(tokens, ends - starts, dim=0)  # 50Hz
    tokens_framewise = torch.repeat_interleave(tokens_framewise, 2)  # 100Hz
    tokens_framewise = tokens_framewise.cpu().numpy()

    # trim to the same length
    minlen = min(len(syllables_framewise), len(tokens_framewise))
    syllables_framewise = syllables_framewise[:minlen]
    tokens_framewise = tokens_framewise[:minlen]

    # ignore all frames that are silences
    voice_activity_mask = syllables_framewise != "SIL"
    syllables_framewise = syllables_framewise[voice_activity_mask]
    tokens_framewise = tokens_framewise[voice_activity_mask]

    # deduplicate for the bitrate calculation
    tokens_deduped = torch.unique_consecutive(torch.from_numpy(tokens_framewise))

    all_syllables_framewise.append(syllables_framewise)
    all_tokens_framewise.append(tokens_framewise)
    all_tokens_deduped.append(tokens_deduped)

all_syllables_framewise = np.concatenate(all_syllables_framewise, axis=0)
all_tokens_framewise = np.concatenate(all_tokens_framewise, axis=0)
all_tokens_deduped = np.concatenate(all_tokens_deduped, axis=0)

syllable_label_encoder = LabelEncoder()
token_label_encoder = LabelEncoder()

all_syllable_ids_duped = syllable_label_encoder.fit_transform(all_syllables_framewise)
all_token_ids_duped = token_label_encoder.fit_transform(all_tokens_framewise)
all_token_ids_deduped = token_label_encoder.transform(all_tokens_deduped)

# ---------- framewise clustering metrics ----------

joint_counts = contingency_matrix(all_syllable_ids_duped, all_token_ids_duped)

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


# ---------- bitrate of the tokens ----------
duration = len(all_syllables_framewise) * 0.01  # (working with 100 Hz)
num_tokens = len(all_tokens_deduped)
token_counts = np.bincount(all_token_ids_deduped)
token_probs = token_counts / token_counts.sum()
token_entropy = -np.sum(token_probs * np.log2(token_probs))
bitrate = num_tokens * token_entropy / duration

print(f"Syllable purity:                        {sp.item():>10.4f}")
print(f"Cluster purity:                         {cp.item():>10.4f}")
print(f"Syllable-normalized mutual information: {snmi.item():>10.4f}")
print(f"Bitrate (over speech frames):           {bitrate:>10.2f}")
