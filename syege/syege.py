import copy
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from sympy import diff, re, simplify
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from rule import get_rule
from symexpr import generate_expression, gen_classification_symbolic, eval_multinomial


def generate_syntetic_rule_based_classifier(n_samples=1000, n_features=2, n_all_features=2, random_state=1, factor=0,
                                            sampling=0.5, explore_domain=False):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0,
                               n_repeated=0, random_state=random_state, n_clusters_per_class=1)

    X0 = copy.deepcopy(X)
    Y0 = copy.deepcopy(y)

    X += factor * np.random.random(size=X.shape)
    X = StandardScaler().fit_transform(X)
    f_min = [X[:, i].min() - sampling*2 for i in range(n_features)]
    f_max = [X[:, i].max() + sampling*2 for i in range(n_features)]

    ff = np.meshgrid(*[np.arange(f_min[i], f_max[i], sampling) for i in range(n_features)], copy=False)

    if explore_domain:
        values = [ff[i].ravel() for i in range(n_features)]
        X_new = np.c_[values].T
    else:
        X_new = X

    knn = KNeighborsClassifier(3)
    knn.fit(X, y)

    Y_new_pp = knn.predict_proba(X_new)[:, 1]
    # Y_new = Y_new.reshape(ff[0].shape)
    # Y_new = Y_new.astype(int)
    Y_new = knn.predict(X_new)

    # y_new = Y_new.ravel()
    dt = DecisionTreeClassifier()
    dt.fit(X_new, Y_new)

    feature_names = ['x%s' % i for i in range(n_all_features)]
    class_name = 'class'
    class_values = [i for i in range(2)]

    def predict_proba(X):
        X = X[:, :n_features]
        return dt.predict_proba(X)

    def predict(X):
        X = X[:, :n_features]
        return dt.predict(X)

    srbc = {
        'dt': dt,
        'ff': ff,
        'X': X_new,
        'Y': Y_new,
        'Ypp': Y_new_pp,
        'feature_names': feature_names,
        'class_name': class_name,
        'class_values': class_values,
        'predict_proba': predict_proba,
        'predict': predict,
        'X0': X0,
        'Y0': Y0,
    }

    return srbc


def get_rule_explanation(x, srbc, n_features, get_values=False):

    dt = srbc['dt']
    feature_names = srbc['feature_names']
    class_name = srbc['class_name']
    class_values = srbc['class_values']
    rule = get_rule(x[:n_features], dt, feature_names, class_name, class_values, feature_names)

    explanation = list()
    rule_premise = defaultdict(float)
    for p in rule.premises:
        sign = 1 if p.op == '>' else -1
        val = sign * p.thr
        rule_premise[p.att] += val

    for feature in sorted(feature_names):
        if not get_values:
            val = 1 if feature in rule_premise else 0
            explanation.append(val)
        else:
            val = rule_premise[feature] if feature in rule_premise else 0.0
            explanation.append(val)

    explanation = np.array(explanation)

    return explanation


def generate_synthetic_linear_classifier(expr=None, n_features=2, n_all_features=2, random_state=1, n_samples=1000,
                                         p_binary=0.7, p_parenthesis=0.3):
    feature_names = None
    if expr is None:
        if random_state is not None:
            np.random.seed(random_state)

        feature_names = ['x%s' % i for i in range(n_all_features)]
        scope = feature_names[:n_features]

        while True:
            # expr = generate_expression(scope, num_operations=num_operations,
            #                            p_binary=p_binary, p_parenthesis=p_parenthesis)
            expr = generate_expression(scope, p_binary=p_binary, p_parenthesis=p_parenthesis)
            expr = str(simplify(expr))
            if np.sum([1 if expr.count(f) > 0 else 0 for f in scope]) == n_features:
                break

    X, Y, Y1 = gen_classification_symbolic(expr, n_samples=n_samples, flip_y=0.0)
    class_name = 'class'
    class_values = [i for i in range(2)]

    evals = Y1
    evals_binary = Y

    evals0 = evals[evals_binary == 0]
    evals1 = evals[evals_binary == 1]

    mm0 = MinMaxScaler(feature_range=(0, 0.5))
    mm0.fit(evals0.reshape(-1, 1))
    mm1 = MinMaxScaler(feature_range=(0.5, 1.0))
    mm1.fit(evals1.reshape(-1, 1))

    def predict_proba(X):
        X = X[:, :n_features]
        evals = list()
        for x in X:
            evals.append(re(eval_multinomial(expr, vals=list(x))))
        evals = np.array(evals)
        evals = np.array(evals, dtype=np.float)
        evals_binary = evals > 0
        evals_binary = evals_binary.flatten()
        evals_binary = np.array(evals_binary, dtype=int)

        evals_scaled = list()
        for x, y in zip(evals, evals_binary):
            if np.isinf(x):
                val = 1.0 if x == +np.inf else 0.0
            elif np.isnan(x):
                val = 0.0
            else:
                if y == 0:
                    val = mm0.transform(x.reshape(-1, 1))[0][0]
                else:
                    val = mm1.transform(x.reshape(-1, 1))[0][0]
                val = max(0.0, min(val, 1.0))
            evals_scaled.append([1.0 - val, val])

        evals_scaled = np.array(evals_scaled)
        return evals_scaled

    def predict(X):
        proba = predict_proba(X)
        return np.argmax(proba, axis=1)

    if feature_names is None:
        feature_names = ['x%s' % i for i in range(n_all_features)]

    slc = {
        'expr': expr,
        'X': X,
        'Y': Y,
        'feature_names': feature_names,
        'class_name': class_name,
        'class_values': class_values,
        'predict_proba': predict_proba,
        'predict': predict,
    }

    return slc


def get_feature_importance_explanation(x, slc, n_features, get_values=True, get_closest_point=False):

    x = x[:n_features]

    expr = slc['expr']
    X = slc['X']
    Y = slc['Y']
    feature_names = slc['feature_names']

    y = np.array(np.array([re(eval_multinomial(expr, vals=list(x)))]) > 0, dtype=int)[0]

    X1 = X[np.where(Y != y)]
    dist1 = cdist(X1, x.reshape(1, -1)).ravel()
    index = np.argsort(dist1)[0]
    cx1 = X1[index]  # closest point to x with different label

    X2 = X[np.where(Y == y)]
    dist2 = cdist(X2, cx1.reshape(1, -1)).ravel() * 0.9 + cdist(X2, x.reshape(1, -1)).ravel() * 0.1
    index = np.argsort(dist2)[0]
    cx = X2[index]  # closet point to cx with same label of x

    explanation = list()
    for i in range(n_features):
        dexpr = diff(expr, 'x%s' % i)
        subs = {'x%s' % fi: v for fi, v in zip(range(n_features), cx)}
        try:
            val = float(re(dexpr.evalf(subs=subs)))
        except TypeError:
            val = 0.0
        val = val if get_values else 1
        explanation.append(val)

    for i in range(n_features, len(feature_names)):
        val = 0.0 if get_values else 0
        explanation.append(val)

    explanation = np.array(explanation)

    if get_closest_point:
        return explanation, cx

    return explanation


def main():

    m = 5
    n = 10

    n_features = 3
    random_state = None

    p_binary = 0.7
    p_parenthesis = 0.3

    slc = generate_synthetic_linear_classifier(n_features=n_features, n_all_features=m, random_state=random_state,
                                               p_binary=p_binary, p_parenthesis=p_parenthesis)

    # slc = generate_synthetic_linear_classifier(expr='x0**2+x1/2', n_features=n_features, n_all_features=m,
    #                                            random_state=random_state, num_operations=num_operations,
    #                                            p_binary=p_binary, p_parenthesis=p_parenthesis)

    expr = slc['expr']
    print(expr)

    X = slc['X']
    Y = slc['Y']
    if slc['feature_names'] is None:
        slc['feature_names'] = ['x%s' % i for i in range(m)]

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))

    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()

    for x in X_test:
        expl_bin = get_feature_importance_explanation(x, slc, n_features, get_values=False)
        expl_val = get_feature_importance_explanation(x, slc, n_features, get_values=True)
        print(x, expl_bin, expl_val)


    # factor = 10
    # sampling = 0.5
    #
    # srbc = generate_syntetic_rule_based_classifier(n_features=n_features, n_all_features=m, random_state=random_state,
    #                                                factor=factor, sampling=sampling)
    #
    # dt = srbc['dt']
    # ff = srbc['ff']
    # X = srbc['X']
    # Y = srbc['Y']
    # feature_names = srbc['feature_names']
    # class_name = srbc['class_name']
    # class_values = srbc['class_values']
    #
    # X_test = np.random.uniform(np.min(ff), np.max(ff), size=(n, m))

    # plt.figure(figsize=(8, 8))
    # cm = plt.cm.PRGn
    # plt.contourf(ff[0], ff[1], Y, cmap=cm, alpha=1)
    # plt.xlim(ff[0].min(), ff[0].max())
    # plt.ylim(ff[1].min(), ff[1].max())
    # # plt.xticks(())
    # # plt.yticks(())
    # plt.show()
    #
    # Y_test = dt.predict(X_test[:, :n_features])
    # plt.figure(figsize=(8, 8))
    # cm = plt.cm.PRGn
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=5, cmap=cm)
    # plt.contourf(ff[0], ff[1], Y, cmap=cm, alpha=0.3)
    # plt.xlim(ff[0].min(), ff[0].max())
    # plt.ylim(ff[1].min(), ff[1].max())
    # # plt.xticks(())
    # # plt.yticks(())
    # plt.show()
    #
    # plt.figure(figsize=(8, 8))
    # cm = plt.cm.PRGn
    # plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), s=5, cmap=cm)
    # plt.contourf(ff[0], ff[1], Y, cmap=cm, alpha=0.3)
    # plt.xlim(ff[0].min(), ff[0].max())
    # plt.ylim(ff[1].min(), ff[1].max())
    # # plt.xticks(())
    # # plt.yticks(())
    # plt.show()

    # for x in X_test:
    #     expl_bin = get_rule_explanation(x, srbc, n_features, get_values=False)
    #     expl_val = get_rule_explanation(x, srbc, n_features, get_values=True)
    #     print(x, expl_bin, expl_val)


if __name__ == "__main__":
    main()

