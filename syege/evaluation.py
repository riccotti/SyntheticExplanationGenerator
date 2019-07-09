import numpy as np
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist


def feature_importance_similarity(a, b, metric='cosine'):
    val = 1.0 - cdist(a.reshape(1, -1), b.reshape(1, -1), metric=metric)[0][0]
    val = max(0.0, min(val, 1.0))
    return val


def rule_based_similarity(a, b):
    return f1_score(a, b, pos_label=1, average='binary')


def word_based_similarity(a, b, use_values=True, use_all_words=True):
    if use_all_words:
        all_words = list(set(a.keys()) | set(b.keys()))
    else:
        all_words = list(b.keys())
    if use_values:
        a_vec = np.array([a.get(w, 0.0) for w in all_words])
        b_vec = np.array([b.get(w, 0.0) for w in all_words])
        return feature_importance_similarity(a_vec, b_vec)
    else:
        a_vec = np.array([1 if w in a else 0 for w in all_words])
        b_vec = np.array([1 if w in b else 0 for w in all_words])
        return rule_based_similarity(a_vec, b_vec)

