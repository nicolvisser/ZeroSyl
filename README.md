# ZeroSyl: Simple Zero-Resource Speech Tokenization for Spoken Language Modeling

[![paper](https://img.shields.io/badge/Paper-Read-%23b31b1b?logo=arXiv&logoColor=%23b31b1b)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
[![bibtex](https://img.shields.io/badge/BibTeX-Cite-008080?style=flat&logo=latex&logoColor=%23008080)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
[![license](https://img.shields.io/badge/Licence-View-blue.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtc2NhbGUtaWNvbiBsdWNpZGUtc2NhbGUiPjxwYXRoIGQ9Ik0xMiAzdjE4Ii8+PHBhdGggZD0ibTE5IDggMyA4YTUgNSAwIDAgMS02IDB6VjciLz48cGF0aCBkPSJNMyA3aDFhMTcgMTcgMCAwIDAgOC0yIDE3IDE3IDAgMCAwIDggMmgyIi8+PHBhdGggZD0ibTUgOCAzIDhhNSA1IDAgMCAxLTYgMHpWNyIvPjxwYXRoIGQ9Ik03IDIxaDEwIi8+PC9zdmc+&logoColor=blue)](./LICENCE)

Spoken language models (a.k.a. pure speech language models) are language models that operate on speech signals. No text is used anywhere in the pipeline. Not during training and neither during inference.

ZeroSyl is a simple method for extracting syllabic units from a WavLM Large model, without requiring to train a complex boundary detector like in previous work.

# Installation

```sh
pip install zerosyl
```

# Basic usage

For continuous embeddings:
```py
from zerosyl import ZeroSylContinuous

model = ZeroSylContinuous.from_remote()
wav = torch.randn(1, 16000)
starts, ends, embeddings = model.encode(wav)
```

For cluster IDs:
```py
from zerosyl import ZeroSylDiscrete

model = ZeroSylDiscrete.from_remote()
wav = torch.randn(1, 16000)
starts, ends, ids = model.encode(wav)
```

For language modeling units:
```py
from zerosyl import ZeroSylCollapsed

model = ZeroSylCollapsed.from_remote()
wav = torch.randn(1, 16000)
starts, ends, ids = model.encode(wav)
```

# Batch encode

To encode large datasets, use the CLI tool and specify a batch size:

```sh
zerosyl encode --batch-size 16 --help
```

# Language model

Our `LanguageModel` is the OPT-125M model.
Refer to the [OPT documentation](https://huggingface.co/docs/transformers/en/model_doc/opt) in the transformers library for control over generation.

```py
from zerosyl import LanguageModel

lm = LanguageModel.from_remote()

# probe likelihoods
brick = torch.tensor([9116, 9115, 3045, 9115])
blick = torch.tensor([9116, 9115, 5041, 9115])
print(lm.loglikelihoods([brick, blick]))

# unconditional generation
print(lm.generate(max_length=10))

```

# Evaluation

Evaluation scripts are also packaged into the CLI tool.

```sh
zerosyl evaluate --help
```

# Method

For those interested in the method, you have several places to start:

1. Read our [paper](https://www.youtube.com/watch?v=dQw4w9WgXcQ).
2. Cloning the repo and working through [explainer.ipynb](explainer.ipynb) and [demo-detect-boundaries.ipynb](demo-detect-boundaries.ipynb).
    ```
    git clone https://github.com/nicolvisser/ZeroSyl/
    ```
3. The core module [zerosyl/zerosyl.py](zerosyl/zerosyl.py) that houses
    - `ZeroSylContinuous` - a wrapper around `WavLM` to add the boundary detection and meanpooling logic.
    - `ZeroSylDiscrete` - a wrapper around `ZeroSylContinuous` to add K-means discretization.
    - `ZeroSylCollapsed` - a wrapper around `ZeroSylDiscrete` to add silence handling.

There is also more information on reproducing the results in the paper inside [reproduce.py](reproduce.py) and [notes/](notes)

