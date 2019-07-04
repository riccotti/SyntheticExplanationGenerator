import numpy as np
import pandas as pd

from lorem import LOREM
from datamanager import prepare_dataset
from util import neuclidean

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
    feature_names = srbc['feature_names']
    class_values = srbc['class_values']
    class_name = srbc['class_name']
    predict_proba = srbc['predict_proba']
    predict = srbc['predict']

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))
    Y_test = predict(X_test)
    data = np.c_[X_test, Y_test]

    df = pd.DataFrame(data=data, columns=feature_names + [class_name])
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name=class_name)

    explainer = LOREM(X_test, predict, feature_names, class_name, class_values, numeric_columns, features_map,
                      neigh_type='geneticp', categorical_use_prob=True, bb_predict_proba=predict_proba,
                      continuous_fun_estimation=True, size=100, ocr=0.1, multi_label=False, one_vs_rest=False,
                      filter_crules=True, random_state=0, verbose=False, ngen=10)

    for i, x in enumerate(X_test):
        print(x)
        exp = explainer.explain_instance(x, samples=1000, use_weights=True, metric=neuclidean)
        expl_val = np.zeros(m).astype(int)
        for c in exp.rule.premises:
            fid = feature_names.index(c.att)
            expl_val[fid] = 1
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

