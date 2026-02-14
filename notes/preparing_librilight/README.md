# Steps in preparing Libri-Light

1. Download from https://github.com/facebookresearch/libri-light
2. Run [Silero VAD](https://github.com/snakers4/silero-vad) to segment the audiobooks into approximately 15 second segments where the split happens at natural pauses in the speech
   - We store the VAD results in a manifest (.parquet file).
   - This file can be opened with pandas.read_parquet
   - They are available to download here
     - https://storage.googleapis.com/zerospeech-checkpoints/librilight-manifest-60k.parquet
     - https://storage.googleapis.com/zerospeech-checkpoints/librilight-manifest-6k-contiguous.parquet
   - One contains VAD segments that are not contiguous. I.e. VAD removes silences. We use this manifest to train the language models for Sylber, ZeroSylCollapsed and ZeroSylDiscrete
   - The other contains VAD segments that are contiguous. I.e. no silence are thrown away, but the splits occur in the middle of natural pauses. We use this manifest to train the langauge model on SyllableLM units. SyllableLM prefers retaining silences.
3. Encode the units with the difference systems
   - For ZeroSyl, using the CLI tool:
    ```
    zerosyl encode --help
    ```
   - For Sylber and SyllableLM, see [encoding_with_other_systems/](../encoding_with_other_systems/)
4. Create a numpy memory map that holds the entire LibriLight training data in one contiguous chunk of memory. This makes it easier to transfer the training data to the node on which the langauge model is trained, and improves the throughput of the dataloader during training.
   - [script_make_librilight_mmap_with_sil_token.py](script_make_librilight_mmap_with_sil_token.py) is used for systems with an explicit single silence token. This script ensures that each VAD segment joins with the next VAD segment with only a single silence token.
   - [script_make_librilight_mmap_without_sil_token.py](script_make_librilight_mmap_with_sil_token.py) is used for systems without an explicit silence token. Here the encoded units of segments are simply concatenated together.

The resulting mmap is used in [train_opt_125m_lm.py](../train_opt125m_lm.py) to train the language model.