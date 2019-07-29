import os
import sys
sys.path.append('../maple')
sys.path.append('../lime')
sys.path.append('../syege')

import datetime
import numpy as np
import pandas as pd

from MAPLE import MAPLE
from shap import KernelExplainer
from lime_tabular import LimeTabularExplainer

from syege import generate_synthetic_linear_classifier
from syege import get_feature_importance_explanation
from evaluation import feature_importance_similarity


def run(black_box, n_records, n_all_features, n_features, random_state, filename):

    n = n_records
    m = n_all_features

    p_binary = 0.7
    p_parenthesis = 0.3

    slc = generate_synthetic_linear_classifier(expr=None, n_features=n_features, n_all_features=m,
                                               random_state=random_state,
                                               p_binary=p_binary, p_parenthesis=p_parenthesis)
    expr = slc['expr']

    X = slc['X']
    feature_names = slc['feature_names']
    class_values = slc['class_values']
    predict_proba = slc['predict_proba']
    # predict = slc['predict']

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))
    Y_test = predict_proba(X_test)[:, 1]

    lime_explainer = LimeTabularExplainer(X_test, feature_names=feature_names, class_names=class_values,
                                          discretize_continuous=False, discretizer='entropy')

    reference = np.zeros(m)
    shap_explainer = KernelExplainer(predict_proba, np.reshape(reference, (1, len(reference))))

    maple_explainer = MAPLE(X_test, Y_test, X_test, Y_test)

    results = list()
    for idx, x in enumerate(X_test):
        gt_val = get_feature_importance_explanation(x, slc, n_features, get_values=True)

        lime_exp = lime_explainer.explain_instance(x, predict_proba, num_features=m)
        # lime_exp_as_dict = {e[0]: e[1] for e in lime_exp.as_list()}
        # lime_expl_val = np.asarray([lime_exp_as_dict.get('x%s' % i, .0) for i in range(m)])
        lime_expl_val = np.array([e[1] for e in lime_exp.as_list()])

        shap_expl_val = shap_explainer.shap_values(x, l1_reg='bic')[1]

        maple_exp = maple_explainer.explain(x)
        maple_expl_val = maple_exp['coefs'][:-1]

        lime_fis = feature_importance_similarity(lime_expl_val, gt_val)
        shap_fis = feature_importance_similarity(shap_expl_val, gt_val)
        maple_fis = feature_importance_similarity(maple_expl_val, gt_val)

        # print(gt_val)
        # print(lime_expl_val)
        # print(shap_expl_val)
        # print(maple_expl_val)

        res = {
            'black_box': black_box,
            'n_records': n_records,
            'n_all_features': n_all_features,
            'n_features': n_features,
            'random_state': random_state,
            'idx': idx,
            'lime': lime_fis,
            'shap': shap_fis,
            'maple': maple_fis,
            'expr': expr,
        }
        results.append(res)
        print(datetime.datetime.now(), 'syege - tlsb', 'black_box %s' % black_box,
              'n_all_features %s' % n_all_features, 'n_features %s' % n_features, 'rs %s' % random_state,
              '%s %s' % (idx, n_records), expr,
              'lime %.2f' % lime_fis, 'shap %.2f' % shap_fis, 'maple %.2f' % maple_fis)

    df = pd.DataFrame(data=results)
    df = df[['black_box', 'n_records', 'n_all_features', 'n_features', 'random_state', 'expr',
             'idx', 'lime', 'shap', 'maple']]
    # print(df.head())

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)


def main():

    n_records = 1000
    n_all_features_list = [256]  #[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    exp_per_naf = 10
    path = '../results/'
    filename = path + 'tabular_linear_synthetic_black_box_256.csv'
    random_state = 0
    max_attempts = 100

    restart = None
    if os.path.isfile(filename):
        restart = pd.read_csv(filename).tail(1).to_dict('record')[0]
        print('restart', restart)

    black_box = 0
    if restart:
        black_box = restart['black_box'] + 1
        random_state = restart['random_state'] + 1
    for n_all_features in n_all_features_list:
        if restart and n_all_features < restart['n_all_features']:
            continue
        if n_all_features == 2:
            n_features_list = [2] * exp_per_naf
        elif n_all_features == 4:
            n_features_list = [2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        elif n_all_features == 8:
            n_features_list = [2, 2, 3, 3, 4, 4, 5, 6, 7, 8]
        else:
            gap = n_all_features / exp_per_naf
            n_features_list_a = np.around(np.arange(2, n_all_features + 1, gap)).astype(int).tolist()
            # n_features_list[-1] = n_all_features
            n_features_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            n_features_list = [x for x in n_features_list if x <= n_all_features]
            n_features_list.extend(n_features_list_a)
            n_features_list = sorted(n_features_list)

        for n_features in n_features_list:
            if restart and n_all_features <= restart['n_all_features'] and n_features <= restart['n_features']:
                continue

            flag = True
            attempts = 0
            while flag and attempts < max_attempts:
                try:
                    print(datetime.datetime.now(), 'syege - tlsb', 'black_box %s' % black_box,
                          'n_all_features %s' % n_all_features, 'n_features %s' % n_features, 'rs %s' % random_state)
                    run(black_box, n_records, n_all_features, n_features, random_state, filename)
                    flag = False
                except ValueError:
                    attempts += 1
                random_state += 1
            black_box += 1


if __name__ == "__main__":
    main()

