# Notes

This directory contains notes on how to reproduce the encodings and langauge models.

## Encoding tokens with other systems

See [encoding_with_other_systems/](encoding_with_other_systems/).

## K-means model

See [train_kmeans.py](train_kmeans.py).

## Evaluating boundaries

To obtain exactly the boundary scores in the paper, we tuned the encoders (during evaluation only) with a constant shift. See [constant_shift/](constant_shift/).

## Preparing and encoding Libri-Light into tokens for the language models

See [preparing_librilight/](./preparing_librilight/).

## Training the OPT-125M language model

See [train_opt125m_lm.py](./train_opt125m_lm.py).