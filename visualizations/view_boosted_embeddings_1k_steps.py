from pathlib import Path

import matplotlib.pyplot as plt
import tgt
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchcodec.decoders import AudioDecoder

from zerosyl.model import ZeroSylBase

root = Path(__file__).parent.parent
checkpoint_path = root / "checkpoints/ZeroSyl-Boost-layer-11-win-3-prom-0_5-steps-1k.pt"
waveforms_dir = root / "data/waveforms/LibriSpeech"
alignments_dir = root / "data/alignments/LibriSpeech"
STEM = "1272-128104-0001"

WAVEFORM_SAMPLE_RATE = 16000
SPECTROGRAM_HOP_LENGTH = 160
s2f = WAVEFORM_SAMPLE_RATE / SPECTROGRAM_HOP_LENGTH

if waveforms_dir.exists() and alignments_dir.exists():
    wav_path = next(waveforms_dir.rglob(f"{STEM}.flac"))
    textgrid_path = next(alignments_dir.rglob(f"{STEM}.TextGrid"))
else:
    # else revert to the sample that is stored in the repository
    wav_path = "data/sample.flac"
    textgrid_path = "data/sample.TextGrid"

textgrid = tgt.read_textgrid(textgrid_path, include_empty_intervals=False)

decoder = AudioDecoder(wav_path, sample_rate=WAVEFORM_SAMPLE_RATE, num_channels=1)
audio = decoder.get_all_samples()

tMelSpectrogram = MelSpectrogram(
    WAVEFORM_SAMPLE_RATE, 1024, 400, SPECTROGRAM_HOP_LENGTH, n_mels=100
)
tAmplitudeToDB = AmplitudeToDB(top_db=80)
melspec = tAmplitudeToDB(tMelSpectrogram(audio.data))[0]

model = ZeroSylBase.from_pretrained_checkpoint(checkpoint_path).cuda()
embeddings = model.framewise_embeddings(audio.data.cuda()).cpu()

# cosine similarity matrix
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
similarity_matrix = embeddings @ embeddings.T
similarity_matrix = similarity_matrix.cpu().numpy()

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10, 10),
    gridspec_kw={"height_ratios": [1, 8], "width_ratios": [1, 8]},
    constrained_layout=True,
)

# --- top left ---
axes[0][0].axis("off")

# --- Top right ---
ax_tr = axes[0][1]
ax_tr.imshow(melspec, aspect="auto", origin="lower")

xticks = []
xtickslabels = []
for interval in textgrid.get_tier_by_name("syllables"):
    x1 = interval.start_time * s2f
    x2 = interval.end_time * s2f
    xticks.append((x1 + x2) / 2)
    xtickslabels.append(interval.text)
    ax_tr.axvline(x1, color="white")
    ax_tr.axvline(x2, color="white")
ax_tr.set_xticks(xticks)
ax_tr.set_xticklabels(xtickslabels, rotation=90)
ax_tr.xaxis.set_ticks_position("top")
ax_tr.xaxis.set_label_position("top")
ax_tr.get_yaxis().set_visible(False)
ax_tr.set_xlim(0, textgrid.end_time * s2f)

# --- Bottom left ---
ax_bl = axes[1][0]
ax_bl.imshow(melspec.T.flip(1), aspect="auto", origin="upper")
yticks = []
ytickslabels = []
for interval in textgrid.get_tier_by_name("syllables"):
    y1 = interval.start_time * s2f
    y2 = interval.end_time * s2f
    yticks.append((y1 + y2) / 2)
    ytickslabels.append(interval.text)
    ax_bl.axhline(y1, color="white")
    ax_bl.axhline(y2, color="white")
ax_bl.set_yticks(yticks)
ax_bl.set_yticklabels(ytickslabels)
ax_bl.get_xaxis().set_visible(False)
ax_tr.set_xlim(0, textgrid.end_time * s2f)

# --- Bottom right ---
ab_br = axes[1][1]
im = ab_br.imshow(similarity_matrix, aspect="equal", origin="upper")
ab_br.axis("off")

plt.show()
