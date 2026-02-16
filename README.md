# ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling

<!-- [![paper](https://img.shields.io/badge/Paper-Read-%23b31b1b?logo=arXiv&logoColor=%23b31b1b)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
[![bibtex](https://img.shields.io/badge/BibTeX-Cite-008080?style=flat&logo=latex&logoColor=%23008080)](https://www.youtube.com/watch?v=dQw4w9WgXcQ) -->
[![license](https://img.shields.io/badge/Licence-View-blue.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgY2xhc3M9Imx1Y2lkZSBsdWNpZGUtc2NhbGUtaWNvbiBsdWNpZGUtc2NhbGUiPjxwYXRoIGQ9Ik0xMiAzdjE4Ii8+PHBhdGggZD0ibTE5IDggMyA4YTUgNSAwIDAgMS02IDB6VjciLz48cGF0aCBkPSJNMyA3aDFhMTcgMTcgMCAwIDAgOC0yIDE3IDE3IDAgMCAwIDggMmgyIi8+PHBhdGggZD0ibTUgOCAzIDhhNSA1IDAgMCAxLTYgMHpWNyIvPjxwYXRoIGQ9Ik03IDIxaDEwIi8+PC9zdmc+&logoColor=blue)](./LICENCE)
[![quickstart](https://img.shields.io/badge/Quick%20start-Open%20in%20Collab-F9AB00?style=flat&logo=googlecolab&logoColor=F9AB00)](https://colab.research.google.com/github/nicolvisser/ZeroSyl/blob/master/quickstart.ipynb)
[![explainer](https://img.shields.io/badge/Explainer-Open%20in%20Collab-F9AB00?style=flat&logo=googlecolab&logoColor=F9AB00)](https://colab.research.google.com/github/nicolvisser/ZeroSyl/blob/master/explainer.ipynb)

Spoken language models (pure speech language models) learn language directly from unlabeled. No text is used anywhere in the pipeline.

ZeroSyl is a simple method for extracting syllable-like units from a WavLM Large model, without requiring to train a complex boundary detector like in previous works.

# Installation

For full functionality (including CLI):

```sh
pip install zerosyl[cli]
```

For base functionality (in other pipelines):

```sh
pip install zerosyl
```

Requires:
- python >=3.11.0,<3.15 (last tested up to 3.14.2)
- torch >=2.4.1,<3.0 (last tested up to 2.10.0)

# Basic usage

[![quickstart](https://img.shields.io/badge/Quick%20start-Open%20in%20Collab-F9AB00?style=flat&logo=googlecolab&logoColor=F9AB00)](https://colab.research.google.com/github/nicolvisser/ZeroSyl/blob/master/quickstart.ipynb)

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
Refer to the [OPT documentation](https://huggingface.co/docs/transformers/en/model_doc/opt) in the transformers library for more functionality including control over generation.

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

1. Our paper will be available on arXiv soon.
2. Working through the [explainer.ipynb](explainer.ipynb) notebook: [![explainer](https://img.shields.io/badge/Explainer-Open%20in%20Collab-F9AB00?style=flat&logo=googlecolab&logoColor=F9AB00)](https://colab.research.google.com/github/nicolvisser/ZeroSyl/blob/master/quickstart.ipynb)
3. Looking at the core module [zerosyl/zerosyl.py](zerosyl/zerosyl.py) that houses
    - `ZeroSylContinuous` - a wrapper around `WavLM` to add the boundary detection and meanpooling logic.
    - `ZeroSylDiscrete` - a wrapper around `ZeroSylContinuous` to add K-means discretization.
    - `ZeroSylCollapsed` - a wrapper around `ZeroSylDiscrete` to add silence handling.

There is also more information on reproducing the results in the paper inside [notes/](notes) and [reproduce.py](reproduce.py).

