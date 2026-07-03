#!/usr/bin/env python3
"""Module that contains the bag_of_words function."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list): A list of the vocabulary words to use for the analysis.
            If None, all words within sentences should be used.

    Returns:
        tuple: (embeddings, features)
        - embeddings: numpy.ndarray of shape (s, f) containing the embeddings.
        - features: numpy.ndarray of the features used for embeddings.
    """
    # Initialize the CountVectorizer with the specified vocabulary
    vectorizer = CountVectorizer(vocabulary=vocab)

    # Fit the vectorizer to the sentences and transform them into a matrix
    X = vectorizer.fit_transform(sentences)

    # Extract the features (vocabulary)
    try:
        # For newer versions of scikit-learn (>= 1.0)
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        # For older versions of scikit-learn
        features = vectorizer.get_feature_names()

    # Convert features to a numpy array to match the expected output format
    features = np.array(features)

    # Return the dense embedding matrix and the features
    return X.toarray(), features
