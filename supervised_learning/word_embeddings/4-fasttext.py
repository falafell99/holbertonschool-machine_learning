#!/usr/bin/env python3
"""Module that contains the fasttext_model function."""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates and trains a gensim fastText model.

    Args:
        sentences (list): A list of sentences to be trained on.
        vector_size (int): The dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word for use.
        window (int): Max distance between current and predicted word.
        negative (int): The size of negative sampling.
        cbow (bool): True is for CBOW; False is for Skip-gram.
        epochs (int): The number of iterations to train over.
        seed (int): The seed for the random number generator.
        workers (int): The number of worker threads to train the model.

    Returns:
        The trained FastText model.
    """
    sg = 0 if cbow else 1
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model
