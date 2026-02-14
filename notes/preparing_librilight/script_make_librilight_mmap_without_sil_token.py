import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


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
        segments = torch.load(f, map_location="cpu").numpy()
        units = segments[:, 2]
        # COUNT THE TOKENS FROM THE UTTERANCE
        n = len(units)
        total_tokens += n

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
        units = segments[:, 2].astype(dtype)

        # ADD THE TOKENS FROM THE UTTERANCE
        n = len(units)
        mm[offset : offset + n] = units
        offset += n

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
        input_dir="/mnt/newt/workspace/zerosyl/output/segments/SylBoost625-k-8192/LibriLightVADContiguous",
        output_prefix="ULM-tokens-LibriLightVADContiguous-6kh-SylBoost625-k-8192",
    )
