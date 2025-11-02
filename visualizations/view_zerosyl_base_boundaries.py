from pathlib import Path

import matplotlib.pyplot as plt
import tgt
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchcodec.decoders import AudioDecoder

from zerosyl.model import ZeroSylBase

root = Path(__file__).parent.parent
checkpoint_path = root / "checkpoints/WavLM-Large.pt"
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
boundaries = model.boundaries(audio.data.cuda())


plt.figure(figsize=(10, 4), constrained_layout=True)
plt.subplot(2, 1, 1)
plt.imshow(melspec, aspect="auto", origin="lower")
xticks = []
xtickslabels = []
for interval in textgrid.get_tier_by_name("syllables"):
    x1 = interval.start_time * s2f
    x2 = interval.end_time * s2f
    xticks.append((x1 + x2) / 2)
    xtickslabels.append(interval.text)
    plt.axvline(x1, color="white")
    plt.axvline(x2, color="white")
plt.xticks(xticks, xtickslabels, rotation=90)
plt.gca().get_yaxis().set_visible(False)
plt.gca().xaxis.set_ticks_position("top")
plt.gca().xaxis.set_label_position("top")
plt.title("Syllables from forced alignments")
plt.xlim(0, textgrid.end_time * s2f)
plt.subplot(2, 1, 2)
plt.imshow(melspec, aspect="auto", origin="lower")
for t in boundaries:
    plt.axvline(t * s2f, color="white")
plt.axis("off")
plt.title("ZeroSyl-Base boundaries")
plt.xlim(0, textgrid.end_time * s2f)
plt.show()
