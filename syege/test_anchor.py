import numpy as np

from anchor.anchor_tabular import AnchorTabularExplainer


from syege import get_rule_explanation, get_rule_explanation_complete
from syege import generate_syntetic_rule_based_classifier
from evaluation import rule_based_similarity, rule_based_similarity_complete


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
    feature_names = srbc['feature_names']
    class_values = srbc['class_values']
    predict_proba = srbc['predict_proba']
    predict = srbc['predict']

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))
    Y_test = predict_proba(X_test)

    explainer = AnchorTabularExplainer(class_names=class_values, feature_names=feature_names, categorical_names={})
    explainer.fit(X_test, Y_test, X_test, Y_test)

    for i, x in enumerate(X_test):
        print(x)
        exp, exp_dict = explainer.explain_instance_ric(x, predict, threshold=0.95)
        # print(exp.features())
        # print(exp_dict)
        # print(exp.exp_map['exp_dict'])
        # break
        expl_val = np.array([1 if e in exp.features() else 0 for e in range(m)])
        gt_val = get_rule_explanation(x, srbc, n_features, get_values=False)
        gt_dict = get_rule_explanation_complete(x, srbc, n_features)
        rbs = rule_based_similarity(expl_val, gt_val)
        rbsc = rule_based_similarity_complete(exp_dict, gt_dict, eps=0.1)
        print(expl_val)
        print(gt_val)
        print(rbs)
        print('')
        print(exp_dict)
        print(gt_dict)
        print(rbsc)
        print('----')

        if i == 10:
            break


if __name__ == "__main__":
    main()

