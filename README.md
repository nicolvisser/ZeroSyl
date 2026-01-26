# ZeroSyl

Simple Zero-Resource Syllable Tokenization for Spoken Langauge Modeling

# Getting started

```bash
git clone https://github.com/nicolvisser/ZeroSyl
cd ZeroSyl
pip install .
```

## Using a notebook

See `demo-quick-start.ipynb`.

## Using the CLI

```bash
zerosyl --help
```

### Encoding

```bash
zerosyl encode --help
```

Encode a single waveform:
```bash
zerosyl encode ./data/sample.flac ./output/sample.pt
```

Encode a directory of waveforms:
```bash
zerosyl encode ./data/waveforms/LibriSpeech/dev-clean ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/dev-clean --extension .flac
```







## Reproducing the evaluations

Download LibriSpeech dev and test sets

```
mkdir -p ./data/waveforms/
wget -O ./data/waveforms/dev-clean.tar.gz https://openslr.trmal.net/resources/12/dev-clean.tar.gz
wget -O ./data/waveforms/dev-other.tar.gz https://openslr.trmal.net/resources/12/dev-other.tar.gz
wget -O ./data/waveforms/test-clean.tar.gz https://openslr.trmal.net/resources/12/test-clean.tar.gz
wget -O ./data/waveforms/test-other.tar.gz https://openslr.trmal.net/resources/12/test-other.tar.gz
```

Encode into segments:

```bash
zerosyl encode ./data/waveforms/LibriSpeech/dev-clean ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/dev-clean --extension .flac
zerosyl encode ./data/waveforms/LibriSpeech/dev-other ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/dev-other --extension .flac
zerosyl encode ./data/waveforms/LibriSpeech/test-clean ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/test-clean --extension .flac
zerosyl encode ./data/waveforms/LibriSpeech/test-other ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/test-other --extension .flac
```

If you want to extract segments for SyllableLM and Sylber you can look at the notebooks `notebooks/syllablelm.ipynb` and `notebooks/sylber.ipynb`.
Since Sylber did not release a clustering model, we trained one.
The centroids can be downloaded with this [link](https://storage.googleapis.com/zerospeech-checkpoints/km-centroids-sylber-k-10000-step-110.npy).

Following Sylber, we allow for a constant shift in the predicted boundaries.
Look at the images at `notebooks/*tune-shift-dev.png` to see the effect of shifting the boundaries.
We pick the following shifts that give the best R-values on the dev sets:

| System           | Constant shift |
| ---------------- | -------------- |
| SylBoost 5.0 Hz  | -0.010 s       |
| SylBoost 6.25 Hz | -0.015 s       |
| SylBoost 8.33 Hz | -0.010 s       |
| Sylber           | -0.040 s       |
| ZeroSyl          | -0.005 s       |

Compute the boundary metrics on the test sets:

```bash
zerosyl eval boundaries ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/ ./data/alignments/LibriSpeech/ --segments-pattern test*/**/*.pt --textgrid-pattern test*/**/*.TextGrid --constant-shift -0.005
```

```
Precision          68.64
Recall             75.38
F1                 71.85
OS                  9.83
R-value            74.57

Token Precision    51.86
Token Recall       56.29
Token F1           53.98
```

Compute the clustering metrics on the dev sets (the test sets give similar scores):

```bash
zerosyl eval clustering ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/ ./data/alignments/LibriSpeech/ --segments-pattern dev*/**/*.pt --textgrid-pattern dev*
/**/*.TextGrid
```

```
Per-cluster purity:                  0.8039
Per-syllable purity:                 0.3184
Syllable-normalized mutual info:     0.8915
```

Compute bitrate and unit frequency on the dev sets:

```bash
zerosyl eval bitrate ./output/segments/ZeroSylCollapsed-v040-k-9116/LibriSpeech/ ./data/alignments/LibriSpeech/ --segments-pattern dev*/**/*.pt --textgrid-pattern dev*/**
/*.TextGrid
```

```
Bitrate:        51.53 bits/s
Frequency:       4.35 Hz
```

