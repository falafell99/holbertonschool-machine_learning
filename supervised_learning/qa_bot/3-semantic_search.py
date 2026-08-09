#!/usr/bin/env python3
"""Module that contains the semantic_search function."""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): the path to the corpus of reference
            documents on which to perform semantic search
        sentence (str): the sentence from which to perform
            semantic search

    Returns:
        str: the reference text of the document most similar to
            sentence
    """
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )

    documents = [sentence]
    filenames = []

    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(corpus_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            documents.append(f.read())
        filenames.append(filepath)

    embeddings = embed(documents)

    sentence_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    correlations = np.inner(sentence_embedding, doc_embeddings)
    closest = np.argmax(correlations)

    with open(filenames[closest], "r", encoding="utf-8") as f:
        return f.read()
