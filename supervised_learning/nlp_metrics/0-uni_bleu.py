#!/usr/bin/env python3
"""Module that contains the uni_bleu function."""
import math


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list): list of reference translations
        sentence (list): list containing the model proposed sentence

    Returns:
        float: the unigram BLEU score
    """
    sen_len = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    closest_len = min(ref_lengths, key=lambda l: (abs(l - sen_len), l))

    counts = {}
    for word in sentence:
        counts[word] = counts.get(word, 0) + 1

    clipped = 0
    for word, count in counts.items():
        max_ref = max(ref.count(word) for ref in references)
        clipped += min(count, max_ref)

    precision = clipped / sen_len

    if sen_len >= closest_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_len / sen_len)

    return bp * precision
