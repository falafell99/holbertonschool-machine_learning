#!/usr/bin/env python3
"""Module that contains the question_answer function."""
semantic_search = __import__('3-semantic_search').semantic_search
answer = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts in a loop.

    Args:
        corpus_path (str): the path to the corpus of reference
            documents
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}
    while True:
        question = input("Q: ")
        if question.lower() in exit_words:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, question)
        result = answer(question, reference)
        if result is None or not result.strip():
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(result))
