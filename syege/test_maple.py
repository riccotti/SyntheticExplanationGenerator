import numpy as np

from MAPLE import MAPLE


from syege import generate_synthetic_linear_classifier
from syege import get_feature_importance_explanation
from evaluation import feature_importance_similarity


def main():
    m = 5
    n = 10

    n_features = 3
    random_state = 2

    p_binary = 0.7
    p_parenthesis = 0.3

    slc = generate_synthetic_linear_classifier(expr=None, n_features=n_features, n_all_features=m,
                                               random_state=random_state, p_binary=p_binary,
                                               p_parenthesis=p_parenthesis)
    expr = slc['expr']
    X = slc['X']
    feature_names = slc['feature_names']
    class_values = slc['class_values']
    predict_proba = slc['predict_proba']
    predict = slc['predict']

    print(expr)

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))
    Y_test = predict(X_test)

    explainer = MAPLE(X_test, Y_test, X_test, Y_test)

    for x in X_test:
        print(x)
        exp = explainer.explain(x)
        expl_val = exp['coefs'][:-1]
        gt_val = get_feature_importance_explanation(x, slc, n_features, get_values=True)
        fis = feature_importance_similarity(expl_val, gt_val)
        print(expl_val)
        print(gt_val)
        print(fis)
        break


if __name__ == "__main__":
    main()
