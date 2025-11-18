# ZeroSpeech Syllable Discovery

## Development

### General rules:
- Don't push to `master`, use branches and pull requests.
- Prepend your username to branches you create e.g. `nicolvisser-feature-x` or `kamperh-patch-1` 
- Keep branches small and focused.
- Check that your code works before pull requests.
- Report any broken code or inaccessible scripts (or fix them).
- Don't let the rules discourage you from contributing. We can always fix things if they break.

### Setup

#### Using poetry

If you plan on making serious changes, such as adding packages or making a new release, then please use [Poetry](https://python-poetry.org/). After it is installed, then `cd` into the cloned repo and run:

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
pip install isort black pre-commit ipykernel torchaudio matplotlib tgt pydub scikit-learn
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

### Downloading the checkpoints and data

We will upload checkpoints and data as releases on GitHub.

Each release will be linked to a commit with a git tag such as `v0.4.0`. The code at that commit will be in a working condition and any required checkpoints or data will be in the Assets section of the corresponding release.

For example you could go to the [releases](https://github.com/nicolvisser/ZeroSyl/releases) page. Then perhaps see that [v0.4.0](https://github.com/nicolvisser/ZeroSyl/tree/v0.4.0) is the latest release. Then checkout the code with:

```
git checkout v0.4.0
```
You will then look at that README for instructions on how to download the required checkpoints and data from the release page.

#### Instructions for v0.4.0:

You should have the following structure inside your workspace:
```
.
├── checkpoints
│   ├── 🟣 WavLM-Large.pt
│   ├── 🔴 km10000-centroids-v040.pt
└── data
    ├── alignments
    │   └── LibriSpeech
    │       ├── 🔴 dev-clean/*
    │       └── 🔴 dev-other/*
    ├── waveforms
    │   └── LibriSpeech
    │       ├── 🟠 dev-clean/*
    │       └── 🟠 dev-other/*
    ├── 🟢 sample.TextGrid
    └── 🟢 sample.flac
```

So go and download the missing thing from the following sources:

| source | link |
|--------|------|
| 🟣     | https://github.com/microsoft/unilm/tree/master/wavlm |
| 🔴     | https://github.com/nicolvisser/ZeroSyl/releases/tag/v0.4.0 |
| 🟠     | https://www.openslr.org/12 |
| 🟢     | Part of repo. No need to download. |

### Demos

See `demo-detect-boundaries.ipynb` and `demo-boosting.ipynb`.

### Evaluations

Running `eval-boundaries.py` should give:

```
Precision: 0.6935, Recall: 0.7564, F1: 0.7236, R-value: 0.7519
Token Precision: 0.5289, Token Recall: 0.5707, Token F1: 0.5490
```

Running `eval-clustering.py` should give:

```
Syllable purity:                            0.6446
Cluster purity:                             0.2487
Syllable-normalized mutual information:     0.8091
Bitrate (over speech frames):                68.66
```

### Formatting

There is a pre-commit hook that might require running the `isort` and `black` formatters.
If your commit fails, just stage the new formatted code and commit again.

If you want to format the code yourself, run:

```bash
poetry run isort .
poery run black .
```

### Poetry releases cheat sheet

Some notes on how to make a new release. Releases are only necessary if we add new checkpoints or data.

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
    ```bash
    git push origin --tags
    ```
6. Create the GitHub Release
    - Go to GitHub → your repo → Releases → “Draft a new release”
    - Then select the tag you created
    - Upload any updated checkpoints + instructionscd 

