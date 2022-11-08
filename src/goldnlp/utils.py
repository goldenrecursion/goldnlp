from collections import defaultdict

import numpy as np
from langdetect import DetectorFactory, detect

from goldnlp.constants import LANGUAGE_CHECK_WINDOW

# make it deterministic as per https://pypi.org/project/langdetect/1.0.7/
DetectorFactory.seed = 0


def adjust_type(type_, *objs):
    if len(objs) == 1:
        return type_(objs[0])
    return [type_(obj) for obj in objs]


def jaccard(A, B):
    A, B = adjust_type(set, A, B)
    return len(A & B) / len(A | B)


def sum_or_union(a, b):
    if isinstance(a, set) and isinstance(b, set):
        return a | b
    return a + b


def logistic_curve(x, L=1, k=1, x_0=0):
    """
    :param x_0: the x-value of the sigmoid's midpoint,
    :param L: the curve's maximum value, and
    :param k: the logistic growth rate or steepness of the curve
    see https://en.wikipedia.org/wiki/Logistic_function
    """
    return L / (1 + np.e ** ((-1 * k) * (x - x_0)))


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector_mean(*vectors):
    return np.array(vectors).mean(axis=0)


def check_language(text, target="en", window=LANGUAGE_CHECK_WINDOW):
    if len(text) < window:
        return True
    text = text[:window]
    return detect(text) == target


def language_detection(text: str):
    tokens = text.split(".")
    lang_counts = defaultdict(int)
    for t in tokens:
        if t:
            lang_counts[detect(t)] += 1
    total_sum = sum(lang_counts.values())
    metrics = {l: c / total_sum for l, c in lang_counts.items()}
    return lang_counts, metrics
