#!/usr/bin/env python3
"""Module that contains the ngram_bleu function."""
import math


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list): list of reference translations
        sentence (list): list containing the model proposed sentence
        n (int): the size of the n-gram to use for evaluation

    Returns:
        float: the n-gram BLEU score
    """
    sen_len = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    closest_len = min(ref_lengths, key=lambda x: (abs(x - sen_len), x))

    sen_ngrams = {}
    for i in range(sen_len - n + 1):
        ngram = tuple(sentence[i:i + n])
        sen_ngrams[ngram] = sen_ngrams.get(ngram, 0) + 1

    clipped = 0
    for ngram, count in sen_ngrams.items():
        max_ref = 0
        for ref in references:
            ref_count = sum(
                1 for i in range(len(ref) - n + 1)
                if tuple(ref[i:i + n]) == ngram
            )
            max_ref = max(max_ref, ref_count)
        clipped += min(count, max_ref)

    total_ngrams = sen_len - n + 1
    if total_ngrams <= 0:
        return 0

    precision = clipped / total_ngrams

    if sen_len >= closest_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_len / sen_len)

    return bp * precision
