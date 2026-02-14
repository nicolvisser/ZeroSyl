# Encoding units using other systems

We tried to use the official extraction pipelines of Sylber and SyllableLM as far as possible.

For SyllableLM, we could use exactly the same units as their full pipeline and checkpoints are available. You can see how we set up their repo and code in `syllablelm.ipynb`.

To setup Sylber, it is as simple as running `pip install sylber`. However, they did not release a clustering model. In other words, the model outputs the boundaries and continuous embeddings, but not discrete IDs. Therefore we trained a clustering model on the embeddings produced by their model. Al the code is documented in `sylber.ipynb`.