from pathlib import Path

import torch
from IPython.display import Audio, display
from neucodec import DistillNeuCodec
from tqdm import tqdm
from zerosyl.cache import BufferCache
from zerosyl.acoustic import AcousticModel
from torch.nn.utils.rnn import pad_sequence

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


loaded_checkpoint = locals().get("loaded_checkpoint", None)
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

semantic_units_list: list[list[int]] = []
for i in range(20):
    segments = torch.load(segments_paths[i])
    semantic_units = segments[:, 2].tolist()
    semantic_units_list.append(semantic_units)


######################## GENERATION PARAMETERS ########################

max_tokens = 1000  # required for cache
max_frames_since_semantic = 20
temperature = 1.0
top_p = 0.85

######################## GENERATION LOGIC ########################

long_kwargs = {"dtype": torch.long, "device": "cuda"}
float_kwargs = {"dtype": torch.float, "device": "cuda"}

assert all(len(x) > 0 for x in semantic_units_list)

batch_size = len(semantic_units_list)
vocab_size = acoustic.output_vocab_size

# encode the prompts
semantic_tokens_list: list[list[int]] = []
encoded_prompts_list: list[list[int]] = []
for semantic_units in semantic_units_list:
    semantic_tokens = [u + acoustic.semantic_offset for u in semantic_units]
    prompt = semantic_tokens + [acoustic.BOS, semantic_tokens[0]]

    semantic_tokens_list.append(semantic_tokens)
    encoded_prompts_list.append(prompt)

# store semantic tokens in a 2D indexable tensor:
semantic_tokens_padded = pad_sequence(
    sequences=[torch.tensor(x, **long_kwargs) for x in semantic_tokens_list],
    batch_first=True,
)  # (batch_size, max(map(len(semantic_tokens_list))))

# create a pointer to the index of the next semantic token to be forced should we encounter an EOU token
cur_semantic_idx = torch.zeros((batch_size,), **long_kwargs)
max_semantic_idx = torch.tensor(list(map(len, semantic_tokens_list)), **long_kwargs) - 1

# compute prompt lengths
promptlens = list(map(len, encoded_prompts_list))  # (batch_size,)

# create cache
cache_window = max(promptlens) + max_tokens
cache = BufferCache(
    n_layers=acoustic.cfg.n_layers,
    max_batch_size=batch_size,
    max_seq_len=cache_window,
    n_kv_heads=acoustic.cfg.n_kv_heads,
    head_dim=acoustic.cfg.head_dim,
)
cache.to(device="cuda", dtype=torch.float32)
cache.reset()

# do a single forward pass on all the prompts
with torch.inference_mode():
    prelogits_batch: torch.Tensor = acoustic(
        torch.tensor(
            sum(encoded_prompts_list, []), **long_kwargs
        ),  # (sum(promptlens),)
        seqlens=promptlens,
        cache=cache,
    )

logits = prelogits_batch.index_select(
    dim=0,
    index=torch.tensor(promptlens, device=prelogits_batch.device).cumsum(dim=0) - 1,
)  # (batch_size, vocab_size)

# free up some memory
del prelogits_batch
torch.cuda.empty_cache()

generated_tokens_list: list[list[int]] = []

# Initialize progress bar
progress_bar = tqdm(total=max([len(s) for s in semantic_units_list]))

# Initialize masks for forcing on the next iteration
force_semantic_next = torch.zeros((batch_size,), dtype=torch.bool, device="cuda")
force_EOS_next = torch.zeros((batch_size,), dtype=torch.bool, device="cuda")

for _ in range(max_tokens):
    # Apply forcing from previous iteration
    if force_semantic_next.any():
        sampled_token = sample(logits, temperature, top_p)
        # Override with semantic tokens where needed
        sampled_token[force_semantic_next] = semantic_tokens_padded[
            force_semantic_next, cur_semantic_idx[force_semantic_next]
        ]
        # Advance the semantic index for items we just forced
        cur_semantic_idx[force_semantic_next] += 1
    else:
        # Normal sampling
        sampled_token = sample(logits, temperature, top_p)

    # Apply EOS forcing
    sampled_token[force_EOS_next] = acoustic.EOS

    # Detect conditions for next iteration
    force_EOS_next = (sampled_token == acoustic.EOS) | (force_EOS_next)
    force_semantic_next = (
        (sampled_token == acoustic.EOU)
        & (cur_semantic_idx < max_semantic_idx)
        & (~force_EOS_next)
    )

    # accumulate the generated tokens in a global list
    generated_tokens_list.append(sampled_token.tolist())

    # stop if all items in batch have reached the end
    if torch.all(force_EOS_next):
        break

    # otherwise make another forward pass using the newly sampled (or forced) tokens
    with torch.inference_mode():
        logits = acoustic(sampled_token, seqlens=[1] * batch_size, cache=cache)

    progress_bar_remaining = progress_bar.total - progress_bar.n
    actual_remaining = (max_semantic_idx - cur_semantic_idx).max().item()
    progress_bar.update(progress_bar_remaining - actual_remaining)
progress_bar.close()

generated_tokens_list = zip(*generated_tokens_list)  # transpose

# decode token sequence to acoustic units

acoustic_tokens_list = []
for generated_tokens in generated_tokens_list:
    acoustic_tokens = [t for t in generated_tokens if t < acoustic.cfg.n_acoustic_types]
    acoustic_tokens_list.append(acoustic_tokens)


# vocode with neucodec
with torch.inference_mode():
    for acoustic_tokens in acoustic_tokens_list:
        acoustic_tokens = torch.tensor(acoustic_tokens, dtype=torch.long, device="cuda")
        waveform = neucodec.decode_code(acoustic_tokens[None, None, :])
        waveform = waveform.squeeze(0)

        display(Audio(data=waveform.cpu(), rate=24000))
