# ZeroSpeech Syllable Discovery

## Development

### General rules:
- Don't push to `master`, use branches.
- Keep branches small and focused.
- Don’t squash other people’s commits.
- Check that your code works before pushing.
- Report any broken code or inaccessible scripts.
- KEEP IT SIMPLE STUPID!

### Setup

#### Using poetry

If you plan on making serious changes, such as adding packages or making a new release, then please use https://python-poetry.org/.

```bash
poetry install
```

This should get everything up and running.

#### Using pip

Alternatively, if you are just checking things out, or want to make minor contributions, then you can install the dependencies with pip.

Install build dependencies:

```bash
pip install -e .
```

Install dev dependencies:

```bash
pip install isort black pre-commit ipykernel torchaudio matplotlib tgt
```

Install training dependencies:

```bash
pip install wandb wandb[media]
```

### Code explanation

In `zerosyl.wavlm` you will find the WavLM's original nn.Module with very minor changes.

In `zerosyl.model` you will find two nn.Modules:
1. `class ZeroSylBase(WavLM):`
    - This is a wrapper around `WavLM` that allows you to conveniently perform PromNormSeg on an intermediate layer and then meanpool a later later within those segments.
    - Look at the `ZeroSylBase.segment(wav)` method this is where the core method is applied.
    -  The rest of the methods are helper and convenience methods.
2. `class ZeroSylDiscrete(ZeroSylBase):`
    - This is a wrapper around ZeroSylBase that adds a codebook and performs vector quantization using cosine distances to get cluster IDs for each segment.
    - Look at `ZeroSylDiscrete.tokenize(wav)` for core functionality.

There is also an nn.Module in `train_boosting_model.py`:
- ``class WavLMWithPredictionHead(WavLM):``
    - This one adds a nn.Linear layer at the end to project to the vocab size. This way we can train WavLM to predict the discovered syllable IDs.

### Visualizations

```
python visualizations/view_raw_embeddings.py
python visualizations/view_zerosyl_base_boundaries.py
python visualizations/view_meanpooled_embeddings.py
python visualizations/view_zerosyl_discrete_tokens.py
python visualizations/view_boosted_embeddings_1k_steps.py
```

In these examples (except the one with discrete tokens) you can play around with `ZeroSylBase`'s hyperparamters by modifying the following attributes right after the model was loaded:

```python
model = ZeroSylBase.from_pretrained_checkpoint(checkpoint_path).cuda()
model.boundary_layer: int = 11  # on which layer to compute norms
model.window_size: int = 3  # for norm smoothing
model.prominence: float = 0.5  # for peak detection
model.meanpool_layer: Optional[int] = None  # which layer to meanpool
```

### Formatting

There is a pre-commit hook that runs the isort and black formatters.
If your commit fails, just stage the new formatted code and commit again.

If you want to format the code yourself, run:

```bash
poetry run isort .
poery run black .
```

### Poetry releases cheat sheet

1. Commit your changes with a descriptive message
2. Update the version using Poetry
    ```bash
    poetry version patch
    ```
    or `poetry version minor` or `poetry version major`
3. Commit the version bump
    ```bash
    git add pyproject.toml
    git commit -m "Bump version to v$(poetry version -s)"
    ```
4. Create a Git tag
    ```bash
    git tag -a v$(poetry version -s) -m "Release v$(poetry version -s)"
    ```
5. Push the changes and tag to GitHub
6. Create the GitHub Release
    - Go to GitHub → your repo → Releases → “Draft a new release”
    - Then select the tag you created
    - Upload any updated checkpoints + instructions


