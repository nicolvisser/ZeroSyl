from pathlib import Path

import tgt
import torchaudio
from tqdm import tqdm

from zerosyl.model import ZeroSylBase
from zerosyl.utils.boundaries import *

TOLERANCE = 0.05  # 50 ms

checkpoint_path = Path("checkpoints/WavLM-Large.pt")
waveform_dir = Path("data/waveforms/LibriSpeech")
alignment_dir = Path("data/alignments/LibriSpeech")

waveform_paths = sorted(waveform_dir.glob("dev*/**/*.flac"))
alignment_paths = sorted(alignment_dir.glob("dev*/**/*.TextGrid"))

assert len(waveform_paths) == len(alignment_paths) == 5567
for wp, ap in zip(waveform_paths, alignment_paths):
    assert wp.stem == ap.stem

model = ZeroSylBase.from_pretrained_checkpoint(checkpoint_path).cuda()

segs, refs = [], []
for waveform_path, alignment_path in zip(tqdm(waveform_paths), alignment_paths):
    waveform, sr = torchaudio.load(str(waveform_path))
    tg = tgt.read_textgrid(alignment_path, include_empty_intervals=True)

    refs.append(
        [float(interval.end_time) for interval in tg.get_tier_by_name("syllables")]
    )
    segs.append(model.boundaries(waveform.cuda()))

# -------------- Calculate boundary evaluation metrics --------------

precision, recall, f1 = eval_boundaries(segs, refs, tolerance=TOLERANCE)

rvalue = get_rvalue(precision, recall)

token_precision, token_recall, token_f1 = eval_token_boundaries(
    segs, refs, tolerance=TOLERANCE
)

print(
    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, R-value: {rvalue:.4f}"
)
print(
    f"Token Precision: {token_precision:.4f}, Token Recall: {token_recall:.4f}, Token F1: {token_f1:.4f}"
)
