#!/usr/bin/env python3
"""Module that contains the cumulative_bleu function."""
import math


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): list of reference translations
        sentence (list): list containing the model proposed sentence
        n (int): the size of the largest n-gram to use for evaluation

    Returns:
        float: the cumulative n-gram BLEU score
    """
    sen_len = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    closest_len = min(ref_lengths, key=lambda x: (abs(x - sen_len), x))

    if sen_len >= closest_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_len / sen_len)

    precisions = []
    for k in range(1, n + 1):
        sen_ngrams = {}
        for i in range(sen_len - k + 1):
            ngram = tuple(sentence[i:i + k])
            sen_ngrams[ngram] = sen_ngrams.get(ngram, 0) + 1

        clipped = 0
        for ngram, count in sen_ngrams.items():
            max_ref = 0
            for ref in references:
                ref_count = sum(
                    1 for i in range(len(ref) - k + 1)
                    if tuple(ref[i:i + k]) == ngram
                )
                max_ref = max(max_ref, ref_count)
            clipped += min(count, max_ref)

        total = sen_len - k + 1
        if total <= 0:
            precisions.append(0)
        else:
            precisions.append(clipped / total)

    if any(p == 0 for p in precisions):
        return 0

    log_avg = sum(math.log(p) for p in precisions) / n
    return bp * math.exp(log_avg)
