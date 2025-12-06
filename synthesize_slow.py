from pathlib import Path

import torch
from IPython.display import Audio, display
from neucodec import DistillNeuCodec
from tqdm import tqdm
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalMask,
)

from zerosyl.acoustic import AcousticModel

# fmt: off

CHECKPOINT_PATH = "checkpoints/ELLA-V-SylBoost625-k-8192-neucodec-LibriSpeech-train-clean-460.pt"
SEGMENTS_DIR = "output/segments/SylBoost625-k-8192/LibriSpeech"

# CHECKPOINT_PATH = "checkpoints/ELLA-V-Sylber-k-10001-neucodec-LibriSpeech-train-clean-460.pt"
# SEGMENTS_DIR = "output/segments/Sylber-k-10001/LibriSpeech"

# CHECKPOINT_PATH = "checkpoints/ELLA-V-ZeroSylCollapsed-v040-k-9116-neucodec-LibriSpeech-train-clean-460.pt"
# SEGMENTS_DIR = "output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech"

# fmt: on


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


loaded_checkpoint = locals().get("loaded_checkpoint_path", None)
if (
    "acoustic" not in locals()
    or "neucodec" not in locals()
    or loaded_checkpoint is None
    or loaded_checkpoint != CHECKPOINT_PATH
):
    acoustic = AcousticModel.from_pretrained_checkpoint(CHECKPOINT_PATH).cuda()
    acoustic.eval()
    loaded_checkpoint = CHECKPOINT_PATH

    neucodec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").cuda()
    neucodec.eval()

segments_dir = Path(SEGMENTS_DIR)
segments_paths = sorted(segments_dir.glob("dev*/**/*.pt"))

semantic_units_batch: list[list[int]] = []

for i in range(3):
    segments = torch.load(segments_paths[i])
    semantic_units = segments[:, 2].tolist()
    semantic_units_batch.append(semantic_units)


max_frames_since_semantic = 20
temperature = 1.0
top_p = 0.85

bsz = len(semantic_units_batch)

input_tokens_batch: list[list[int]] = []
semantic_tokens_remaining_batch: list[list[int]] = []
for semantic_units in semantic_units_batch:
    semantic_tokens = [u + acoustic.semantic_offset for u in semantic_units]
    prompt = semantic_tokens + [acoustic.BOS, semantic_tokens[0]]
    input_tokens_batch.append(prompt)
    semantic_tokens_remaining_batch.append(semantic_tokens[1:])

reached_EOS_batch = [False] * bsz
frames_since_semantic_batch = [0] * bsz
force_semantic_batch = [False] * bsz
acoustic_tokens_batch: list[list[int]] = [None] * bsz

# progress tracking
max_num_semantic_tokens = max([len(s) for s in semantic_units_batch])
pbar = tqdm(total=max_num_semantic_tokens)
prev_processed = 0


while sum(reached_EOS_batch) < bsz:

    input_tokens_batch_flat = sum(input_tokens_batch, [])
    input_tokens_batch_flat = torch.tensor(
        input_tokens_batch_flat, dtype=torch.long, device="cuda"
    )
    seqlens = [len(p) for p in input_tokens_batch]

    with torch.inference_mode():
        logits = acoustic.forward(input_tokens_batch_flat, seqlens)

    last_token_indices = (
        torch.cumsum(torch.tensor(seqlens, dtype=torch.long, device="cuda"), dim=0) - 1
    )
    last_token_logits_batch = logits.index_select(dim=0, index=last_token_indices)
    del logits
    torch.cuda.empty_cache()

    predicted_token_batch = sample(last_token_logits_batch, temperature, top_p).tolist()

    for b in range(bsz):
        if reached_EOS_batch[b]:
            pass
        elif (
            force_semantic_batch[b]
            or frames_since_semantic_batch[b] >= max_frames_since_semantic
        ):
            if len(semantic_tokens_remaining_batch[b]) <= 0:
                reached_EOS_batch[b] = True
                if acoustic_tokens_batch[b] is None:
                    acoustic_tokens_batch[b] = [
                        t
                        for t in input_tokens_batch[b]
                        if t < acoustic.cfg.n_acoustic_types
                    ]
                input_tokens_batch[b] = [0]  # dummy sequence to save compute
            else:
                input_tokens_batch[b] += [semantic_tokens_remaining_batch[b].pop(0)]
                frames_since_semantic_batch[b] = 0
        else:
            input_tokens_batch[b] += [predicted_token_batch[b]]
            frames_since_semantic_batch[b] += 1

    force_semantic_batch = [t == acoustic.EOU for t in predicted_token_batch]

    processed = max_num_semantic_tokens - max(
        [len(s) for s in semantic_tokens_remaining_batch]
    )
    if processed > prev_processed:
        pbar.update(processed - prev_processed)
        prev_processed = processed
pbar.close()


# vocode with neucodec
with torch.inference_mode():
    for acoustic_tokens in acoustic_tokens_batch:
        acoustic_tokens = torch.tensor(acoustic_tokens, dtype=torch.long, device="cuda")
        waveform = neucodec.decode_code(acoustic_tokens[None, None, :])
        waveform = waveform.squeeze(0)

        display(Audio(data=waveform.cpu(), rate=24000))
