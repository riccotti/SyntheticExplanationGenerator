import numpy as np

from shap import TreeExplainer


from syege import get_rule_explanation
from syege import generate_syntetic_rule_based_classifier
from evaluation import rule_based_similarity


def main():
    m = 5
    n = 10

    n_features = 2
    random_state = 1

    factor = 10
    sampling = 0.5

    srbc = generate_syntetic_rule_based_classifier(n_features=n_features, n_all_features=m, random_state=random_state,
                                                   factor=factor, sampling=sampling)

    X = srbc['X']
    dt = srbc['dt']
    predict_proba = srbc['predict_proba']

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))
    Y_test = predict_proba(X_test)

    explainer = TreeExplainer(dt)

    for i, x in enumerate(X_test):
        print(x)
        shap_values = explainer.shap_values(x)[1]
        expl_val = np.array([1 if e != 0.0 else 0 for e in shap_values])
        gt_val = get_rule_explanation(x, srbc, n_features, get_values=False)
        rbs = rule_based_similarity(expl_val, gt_val)
        print(expl_val)
        print(gt_val)
        print(rbs)
        print('')
        if i == 10:
            break


if __name__ == "__main__":
    main()


