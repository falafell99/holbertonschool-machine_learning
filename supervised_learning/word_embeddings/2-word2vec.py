#!/usr/bin/env python3
"""Module that contains the word2vec_model function."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model.

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
        The trained Word2Vec model.
    """
    # Map cbow boolean to sg parameter (0 for CBOW, 1 for Skip-gram)
    sg = 0 if cbow else 1

    try:
        # Gensim 4.0.0 and above
        model = gensim.models.Word2Vec(sentences=sentences,
                                       vector_size=vector_size,
                                       min_count=min_count,
                                       window=window,
                                       negative=negative,
                                       sg=sg,
                                       epochs=epochs,
                                       seed=seed,
                                       workers=workers)
    except TypeError:
        # Fallback for older Gensim versions (< 4.0.0) where parameters
        # 'vector_size' and 'epochs' were called 'size' and 'iter'
        model = gensim.models.Word2Vec(sentences=sentences,
                                       size=vector_size,
                                       min_count=min_count,
                                       window=window,
                                       negative=negative,
                                       sg=sg,
                                       iter=epochs,
                                       seed=seed,
                                       workers=workers)

    return model
