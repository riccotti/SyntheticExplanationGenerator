import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.spatial.distance import cdist
from collections import defaultdict


def generate_rw(y0=None, m=128, mu=0.0, sigma=1.0, ymin=None, ymax=None):
    y0 = y0 if y0 is not None else np.random.normal(mu, sigma)
    rw = np.zeros(m)
    rw[0] = y0
    for i in range(1, m):
        yt = rw[i-1] + np.random.normal(mu, sigma)
        if (ymax is not None) and (yt > ymax):
            yt = rw[i-1] - abs(np.random.normal(mu, sigma))
        elif (ymin is not None) and (yt < ymin):
            yt = rw[i-1] + abs(np.random.normal(mu, sigma))
        rw[i] = yt
    return rw


def generate_ts_dataset(n=10, m=128, ymin=-10, ymax=10):
    X = list()
    for _ in range(n):
        y0 = np.random.randint(ymin, ymax + 1)
        X.append(generate_rw(y0=y0, m=m, mu=0.0, sigma=1.0, ymin=ymin, ymax=ymax))
    X = np.array(X)
    return X


def generate_boundaries(m=128, n_classes=2, k_min=4, k_max=8, l_min=4, l_max=16, mu=0.0, sigma=1.0):
    Sd = dict()
    for c in range(n_classes):
        S = list()
        k = np.random.randint(k_min, k_max + 1)
        for ki in range(k):
            l = np.random.randint(l_min, l_max + 1)
            s = np.random.randint(0, m - l + 1)
            w = np.random.normal(mu, sigma)
            S.append((s, l, w))
        Sd[c] = S

    return Sd


def generate_subsequences(n_classes=2, ymin=-10, ymax=10, k_min=4, k_max=8, l_min=4, l_max=16,
                          mu_min=-2, mu_max=2, sigma_min=0.0, sigma_max=2.5, sigma_gap=0.5):
    Sd = dict()
    for c in range(n_classes):

        S = list()
        k = np.random.randint(k_min, k_max + 1)
        for ki in range(k):
            y0 = np.random.randint(ymin, ymax + 1)
            l = np.random.randint(l_min, l_max + 1)
            mu_s = np.random.randint(mu_min, mu_max + 1)
            sigma_s = np.random.choice(np.arange(sigma_min, sigma_max + sigma_gap, sigma_gap))

            s = generate_rw(y0=y0, m=l, mu=mu_s, sigma=sigma_s, ymin=ymin, ymax=ymax)
            S.append(s)

        Sd[c] = S

    return Sd


def plot_dataset_boundaries(X, Sd, ts_color='k'):
    color_map = {i: v for i, v in enumerate(mcolors.TABLEAU_COLORS.values())}
    plt.plot(X.T, c=ts_color)
    for c, S in Sd.items():
        for slw in S:
            plt.axvspan(slw[0], slw[0] + slw[1], alpha=0.5, color=color_map[c])
    plt.show()


def plot_dataset_subsequences(X, Sd, ts_color='k'):
    color_map = {i: v for i, v in enumerate(mcolors.TABLEAU_COLORS.values())}
    plt.plot(X.T, c=ts_color)
    for c, S in Sd.items():
        for s in S:
            plt.plot(s, c=color_map[c])
    plt.show()


def generate_synthetic_ts_classifier_b(m=128, n_classes=2, k_min=4, k_max=8, l_min=4, l_max=16,
                                       mu=0.0, sigma=1.0, random_state=None):
    if random_state:
        np.random.seed(random_state)

    Sd = generate_boundaries(m, n_classes, k_min, k_max, l_min, l_max, mu, sigma)

    def predict_proba(X):
        n = X.shape[0]
        C = np.zeros((n, n_classes))
        for c, S in Sd.items():
            for i, x in enumerate(X):
                vals = list()
                for j, slw in enumerate(S):
                    vals.append(slw[2] * np.mean(x[slw[0]:slw[0] + slw[1]]))

                C[i, c] = np.mean(np.abs(vals))

        for i in range(len(X)):
            C[i] = C[i] / np.sum(C[i])

        proba = np.array(C)
        return proba

    def predict(X):
        proba = predict_proba(X)
        return np.argmax(proba, axis=1)

    srbc = {
        'n_classes': n_classes,
        'k_min': k_min,
        'k_max': k_max,
        'l_min': l_min,
        'l_max': l_max,
        'mu': mu,
        'sigma': sigma,
        'predict_proba': predict_proba,
        'predict': predict,
        'boundaries': Sd,
    }

    return srbc


def generate_synthetic_ts_classifier_s(n_classes=2, ymin=-10, ymax=10, k_min=4, k_max=8, l_min=4, l_max=16,
                                       mu_min=-2, mu_max=2, sigma_min=0.0, sigma_max=2.5, sigma_gap=0.5, p=50,
                                       random_state=None):
    if random_state:
        np.random.seed(random_state)

    Sd = generate_subsequences(n_classes, ymin, ymax, k_min, k_max, l_min, l_max,
                               mu_min, mu_max, sigma_min, sigma_max, sigma_gap)

    def predict_proba(X):

        n = X.shape[0]
        C = np.zeros((n, n_classes))
        for c, S in Sd.items():
            D = np.zeros((len(X), len(S)))
            for i, x in enumerate(X):
                for j, s in enumerate(S):
                    D[i, j] = dist(s, x)
            thrs = np.percentile(D, p, axis=0)

            for i in range(len(X)):
                C[i][c] += np.sum(D[i] <= thrs)
            C[:, c] /= len(Sd[c])

        for i in range(len(X)):
            C[i] = C[i] / np.sum(C[i])

        proba = np.array(C)
        return proba

    def predict(X):
        proba = predict_proba(X)
        return np.argmax(proba, axis=1)

    srbc = {
        'n_classes': n_classes,
        'ymin': ymin,
        'ymax': ymax,
        'k_min': k_min,
        'k_max': k_max,
        'l_min': l_min,
        'l_max': l_max,
        'mu_min': mu_min,
        'mu_max': mu_max,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'sigma_gap': sigma_gap,
        'predict_proba': predict_proba,
        'predict': predict,
        'subsequences': Sd
    }

    return srbc


def get_boundary_explanation(x, boundaries):

    C = np.zeros(len(boundaries))
    V = dict()
    for c, S in boundaries.items():
        vals = list()
        for j, slw in enumerate(S):
            vals.append(slw[2] * np.mean(x[slw[0]:slw[0] + slw[1]]))

        C[c] = np.mean(np.abs(vals))
        V[c] = np.abs(vals)

    y = np.argmax(C)
    exp = list()
    for i, slw in enumerate(boundaries[y]):
        exp.append((slw[0], slw[1], V[y][i]))

    return exp


def get_subsequence_explanation(x, subsequences, p=50):

    C = np.zeros(len(subsequences))
    E = dict()
    Idx = dict()
    for c, S in subsequences.items():
        D = np.zeros(len(S))
        I = np.zeros(len(S))
        for j, s in enumerate(S):
            D[j], I[j] = dist(s, x, get_idx=True)
        thrs = np.percentile(D, p, axis=0)

        E[c] = D <= thrs
        C[c] = np.sum(D <= thrs) / len(subsequences[c])
        Idx[c] = I

    y = np.argmax(C)
    exp = list()
    for i, s in enumerate(subsequences[y]):
        if E[y][i]:
            exp.append((s, int(Idx[y][i])))

    return exp


def dist(s, x, get_idx=False):
    m = len(s)
    values = list()
    for i in range(len(x) - m):
        values.append(cdist(s.reshape(1, -1), x[i:i + m].reshape(1, -1))[0][0])

    if get_idx:
        return np.min(values), int(np.argmin(values))

    return np.min(values)


def main():
    n = 100
    m = 128
    ymin = -10
    ymax = 10
    n_classes = 2
    k_min = 4  # min nbr of subsequences
    k_max = 8  # max nbr of subsequences
    l_min = 4  # min length
    l_max = 16  # max length
    p = 50  # thr closeness subsequences
    mu_min = -2
    mu_max = 2
    sigma_min = 0.0
    sigma_max = 2.5
    sigma_gap = 0.5
    mu = 0.0
    sigma = 1.0
    random_state = 0

    stc_s = generate_synthetic_ts_classifier_s(n_classes, ymin, ymax, k_min, k_max, l_min, l_max,
                                               mu_min, mu_max, sigma_min, sigma_max, sigma_gap, p, random_state)

    predict = stc_s['predict']
    predict_proba = stc_s['predict_proba']
    subsequences = stc_s['subsequences']

    X = generate_ts_dataset(n, m, ymin, ymax)

    y = predict(X)
    print(np.unique(y, return_counts=True))

    idx = 0
    x = X[idx]
    print(x)
    print(y[idx])
    # plt.title('b(x) = %s' % y[idx])
    # plt.plot(x, c='k')
    # plt.show()

    exp = get_subsequence_explanation(x, subsequences)
    print(exp)

    # plt.title('b(x) = %s' % y[idx])
    # plt.plot(x, c='k')
    # for e in exp:
    #     s, idx = e
    #     plt.plot(range(idx, idx + len(s)), s, c='g')
    # plt.show()

    sts_b = generate_synthetic_ts_classifier_b(m, n_classes, k_min, k_max, l_min, l_max, mu, sigma, random_state=None)

    predict = sts_b['predict']
    predict_proba = sts_b['predict_proba']
    boundaries = sts_b['boundaries']

    y = predict(X)
    print(np.unique(y, return_counts=True))
    idx = 0
    x = X[idx]
    print(x)
    print(y[idx])
    # plt.title('b(x) = %s' % y[idx])
    # plt.plot(x, c='k')
    # plt.show()

    exp = get_boundary_explanation(x, boundaries)
    print(exp)

    # plt.title('b(x) = %s' % y[idx])
    # plt.plot(x, c='k')
    # all_w = [e[2] for e in exp]
    # tot_w = np.sum(np.abs(all_w))
    # for e in exp:
    #     s, l, w = e
    #     plt.axvspan(s, s + l, alpha=np.abs(w)/tot_w, color=color_map[c])
    # plt.show()


if __name__ == "__main__":
    main()
