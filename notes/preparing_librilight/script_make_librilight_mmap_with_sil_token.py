import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SILENCE_TOKEN_ID = 9115


def build_memmap(input_dir, output_prefix, dtype=np.uint16):
    input_dir = Path(input_dir)
    out_data = Path(f"{output_prefix}.bin")
    out_meta = Path(f"{output_prefix}.json")

    print("Scanning input directory...")
    files = []
    files.extend(sorted(input_dir.glob("small/**/*.pt")))
    files.extend(sorted(input_dir.glob("medium/**/*.pt")))
    # files.extend(sorted(input_dir.glob("large/**/*.pt")))
    print(f"Found {len(files)} files.")

    # --------------------------------------------------
    # 1. Determine total number of tokens
    # --------------------------------------------------
    print("Counting total tokens...")
    total_tokens = 0

    for i, f in enumerate(tqdm(files)):
        # GET UNITS FROM THE SEGMENT FILE
        segments = torch.load(f, map_location="cpu")
        units = segments[:, 2]

        if len(units) <= 2:
            continue

        # STRIP THE SILENCE TOKEN IF THERE IS ONE AT START
        if units[0] == SILENCE_TOKEN_ID:
            units = units[1:]
        # STRIP THE SILENCE TOKEN IF THERE IS ONE AT END
        if units[-1] == SILENCE_TOKEN_ID:
            units = units[:-1]

        # COUNT A SILENCE TOKEN AT THE START
        total_tokens += 1

        # COUNT THE TOKENS FROM THE UTTERANCE
        n = units.numel()
        total_tokens += n

        # COUNT THE TOKEN AT THE END OF THE DATASET
        if i == len(files) - 1:
            total_tokens += 1

    print(f"Total tokens: {total_tokens:,}")

    # --------------------------------------------------
    # 2. Create the memmap file of the correct size
    # --------------------------------------------------
    print("Allocating memmap file...")
    mm = np.memmap(out_data, dtype=dtype, mode="w+", shape=(total_tokens,))

    # --------------------------------------------------
    # 3. Stream-write all tokens
    # --------------------------------------------------
    print("Writing tokens into memmap...")
    offset = 0
    for i, f in enumerate(tqdm(files)):
        # GET UNITS FROM THE SEGMENT FILE
        segments = torch.load(f, map_location="cpu").numpy()
        units = segments[:, 2]

        if len(units) <= 2:
            continue

        # STRIP THE SILENCE TOKEN IF THERE IS ONE AT START
        if units[0] == SILENCE_TOKEN_ID:
            units = units[1:]
        # STRIP THE SILENCE TOKEN IF THERE IS ONE AT END
        if units[-1] == SILENCE_TOKEN_ID:
            units = units[:-1]

        # ALWAYS ADD A SILENCE TOKEN AT THE START
        mm[offset] = SILENCE_TOKEN_ID
        offset += 1

        # ADD THE TOKENS FROM THE UTTERANCE
        n = len(units)
        mm[offset : offset + n] = units
        offset += n

        # ADD THE TOKEN AT THE END OF THE DATASET
        if i == len(files) - 1:
            mm[offset] = SILENCE_TOKEN_ID
            offset += 1

    print("Flushing memmap...")
    mm.flush()

    assert offset == total_tokens

    # --------------------------------------------------
    # 4. Write metadata
    # --------------------------------------------------
    meta = {
        "dtype": "uint16",
        "total_tokens": int(total_tokens),
    }
    with open(out_meta, "w") as fp:
        json.dump(meta, fp, indent=2)

    print("Done!")
    print(f"Data file: {out_data}")
    print(f"Metadata:  {out_meta}")


if __name__ == "__main__":
    build_memmap(
        input_dir="/mnt/newt/zerosyl/output/segments/ZeroSylCollapsed-v040-k-9116/LibriLightVAD",
        output_prefix="ULM-tokens-LibriLight-6kh-ZeroSylCollapsed-v040-k-9116",
    )
