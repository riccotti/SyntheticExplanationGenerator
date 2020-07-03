import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from scipy.spatial.distance import cdist


def feature_importance_similarity(a, b, metric='cosine'):
    val = 1.0 - cdist(a.reshape(1, -1), b.reshape(1, -1), metric=metric)[0][0]
    val = max(0.0, min(val, 1.0))
    return val


def rule_based_similarity(a, b):
    return f1_score(a, b, pos_label=1, average='binary')


def rule_based_precision(a, b):
    return precision_score(a, b, pos_label=1, average='binary')


def rule_based_recall(a, b):
    return recall_score(a, b, pos_label=1, average='binary')


def word_based_similarity(a, b, use_values=True, ret_pre_rec=False):
    a_indexs = np.where(a != 0)[0]
    b_indexs = np.where(b != 0)[0]
    all_indexes = sorted(set(list(a_indexs)) | set(list(b_indexs)))
    a_vec = np.array([a[i] for i in all_indexes])
    b_vec = np.array([b[i] for i in all_indexes])
    if use_values:
        return feature_importance_similarity(a, b)
    else:
        a_vec = np.array([1 if v != 0 else 0 for v in a_vec])
        b_vec = np.array([1 if v != 0 else 0 for v in b_vec])
        if ret_pre_rec:
            pre = rule_based_precision(a_vec, b_vec)
            rec = rule_based_recall(a_vec, b_vec)
            f1 = rule_based_similarity(a_vec, b_vec)
            return f1, pre, rec
        return rule_based_similarity(a_vec, b_vec)


def word_based_similarity_text(a, b, use_values=True, use_all_words=False, ret_pre_rec=False):
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
        if ret_pre_rec:
            pre = rule_based_precision(a_vec, b_vec)
            rec = rule_based_recall(a_vec, b_vec)
            f1 = rule_based_similarity(a_vec, b_vec)
            return f1, pre, rec
        return rule_based_similarity(a_vec, b_vec)


def pixel_based_similarity(a, b, ret_pre_rec=False):
    if np.sum(a) == 0.0 and np.sum(b) == 0.0:
        if ret_pre_rec:
            return 1.0, 1.0, 1.0
        return 1.0

    if ret_pre_rec:
        pre = rule_based_precision(a, b)
        rec = rule_based_recall(a, b)
        f1 = rule_based_similarity(a, b)
        return f1, pre, rec

    return rule_based_similarity(a, b)


def rule_based_similarity_complete(a, b, eps=0.01):
    score = 0.0
    features = set(a.keys() | b.keys())
    den = 0
    for f in features:
        default = np.inf if f[1] == '<=' else -np.inf

        v_a = a[f] if f in a else default
        v_b = b[f] if f in a else default

        if (v_a == v_b and v_a == np.inf) or (v_a == v_b and v_a == -np.inf):
            continue
        den += 1
        if np.abs(v_a - v_b) <= eps:
            val = 1.0
        else:
            val = 0.0

        print(f, v_a, v_b, np.abs(v_a - v_b), val)
        score += val

    score = score / den
    return score
