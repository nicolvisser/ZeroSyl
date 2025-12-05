from pathlib import Path

import torch
from IPython.display import Audio, display
from neucodec import DistillNeuCodec

from zerosyl.acoustic import AcousticModel

acoustic = AcousticModel.from_pretrained_checkpoint(
    "wandb/run-20251205_081358-ua4bc3k4/files/best.pt"
).cuda()

neucodec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").cuda()
neucodec.eval()

# load semantic units from data stored as segments
segments_dir = Path("output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech")
segments_path = sorted(segments_dir.glob("dev*/**/*.pt"))[200]
segments = torch.load(segments_path)
semantic_units = segments[:, 2]

# generate acoustic units
acoustic_units = acoustic.generate(
    semantic_units, top_p=0.85, max_tokens_per_semantic_unit=20
)

# vocode with neucodec
with torch.inference_mode():
    waveform = neucodec.decode_code(acoustic_units[None, None, :].cuda())
waveform = waveform.squeeze(0)

display(Audio(data=waveform.cpu(), rate=24000))
