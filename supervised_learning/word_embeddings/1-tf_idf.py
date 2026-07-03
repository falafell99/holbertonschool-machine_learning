#!/usr/bin/env python3
"""Module that contains the tf_idf function."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list): A list of the vocabulary words to use for the analysis.
            If None, all words within sentences should be used.

    Returns:
        tuple: (embeddings, features)
        - embeddings: numpy.ndarray of shape (s, f) containing the embeddings.
        - features: numpy.ndarray of the features used for embeddings.
    """
    # Initialize the TfidfVectorizer with the specified vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit the vectorizer to the sentences and transform them
    X = vectorizer.fit_transform(sentences)

    # Extract the features (vocabulary)
    try:
        # For newer versions of scikit-learn (>= 1.0)
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        # For older versions of scikit-learn
        features = vectorizer.get_feature_names()

    # Convert features to a numpy array
    features = np.array(features)

    # Return the dense embedding matrix and the features
    return X.toarray(), features
