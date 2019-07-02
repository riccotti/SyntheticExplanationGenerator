from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist


def feature_importance_similarity(a, b, metric='cosine'):
    val = 1.0 - cdist(a.reshape(1, -1), b.reshape(1, -1), metric=metric)[0][0]
    val = max(0.0, min(val, 1.0))
    return val


def rule_based_similarity(a, b):
    return f1_score(a, b, pos_label=1, average='binary')

