import os
import sys
sys.path.append('../maple')
sys.path.append('../lime')
sys.path.append('../syege')

import scipy.special
import datetime
import numpy as np
import pandas as pd

from MAPLE import MAPLE
from shap import KernelExplainer
from lime_tabular import LimeTabularExplainer

from syege import generate_synthetic_linear_classifier2
from syege import get_feature_importance_explanation2
from evaluation import feature_importance_similarity, rule_based_similarity


def run(black_box, n_records, n_all_features, n_features, n_coefficients, random_state, filename):

    n = n_records
    m = n_all_features

    slc = generate_synthetic_linear_classifier2(n_features=n_features, n_all_features=n_all_features,
                                                n_coefficients=n_coefficients,
                                                random_state=random_state)

    feature_names = slc['feature_names']
    class_values = slc['class_values']
    predict_proba = slc['predict_proba']
    # predict = slc['predict']

    X_test = np.random.uniform(size=(n, n_all_features))
    Xz = list()
    for x in X_test:
        nz = np.random.randint(0, n_features)
        zeros_idx = np.random.choice(np.arange(n_features), size=nz, replace=False)
        x[zeros_idx] = 0.0
        Xz.append(x)
    X_test = np.array(Xz)

    Y_test = predict_proba(X_test)[:, 1]

    lime_explainer = LimeTabularExplainer(X_test, feature_names=feature_names, class_names=class_values,
                                          discretize_continuous=False, discretizer='entropy')

    reference = np.zeros(m)
    shap_explainer = KernelExplainer(predict_proba, np.reshape(reference, (1, len(reference))))

    maple_explainer = MAPLE(X_test, Y_test, X_test, Y_test)

    results = list()
    for idx, x in enumerate(X_test):
        gt_val = get_feature_importance_explanation2(x, slc, n_features, n_all_features, get_values=True)
        gt_val_bin = get_feature_importance_explanation2(x, slc, n_features, n_all_features, get_values=False)

        lime_exp = lime_explainer.explain_instance(x, predict_proba, num_features=m)
        lime_expl_val = np.array([e[1] for e in lime_exp.as_list()])
        tmp = np.zeros(lime_expl_val.shape)
        tmp[np.where(lime_expl_val != 0.0)] = 1.0
        lime_expl_val_bin = tmp

        shap_expl_val = shap_explainer.shap_values(x, l1_reg='bic')[1]
        tmp = np.zeros(shap_expl_val.shape)
        tmp[np.where(shap_expl_val != 0.0)] = 1.0
        shap_expl_val_bin = tmp

        maple_exp = maple_explainer.explain(x)
        maple_expl_val = maple_exp['coefs'][:-1]
        tmp = np.zeros(maple_expl_val.shape)
        tmp[np.where(maple_expl_val != 0.0)] = 1.0
        maple_expl_val_bin = tmp

        lime_fis = feature_importance_similarity(lime_expl_val, gt_val)
        shap_fis = feature_importance_similarity(shap_expl_val, gt_val)
        maple_fis = feature_importance_similarity(maple_expl_val, gt_val)

        lime_rbs = rule_based_similarity(lime_expl_val_bin, gt_val_bin)
        shap_rbs = rule_based_similarity(shap_expl_val_bin, gt_val_bin)
        maple_rbs = rule_based_similarity(maple_expl_val_bin, gt_val_bin)

        # print(gt_val)
        # print(lime_expl_val)
        # print(shap_expl_val)
        # print(maple_expl_val)

        res = {
            'black_box': black_box,
            'n_records': n_records,
            'n_all_features': n_all_features,
            'n_features': n_features,
            'n_coefficients': n_coefficients,
            'random_state': random_state,
            'idx': idx,
            'lime_cs': lime_fis,
            'shap_cs': shap_fis,
            'maple_cs': maple_fis,
            'lime_f1': lime_rbs,
            'shap_f1': shap_rbs,
            'maple_f1': maple_rbs,
        }
        results.append(res)
        print(datetime.datetime.now(), 'syege - tlsb2', 'black_box %s' % black_box,
              'n_all_features %s' % n_all_features, 'n_features %s' % n_features,
              'n_coefficients % s' % n_coefficients,
              'rs %s' % random_state,
              '%s %s' % (idx, n_records),
              'lime %.2f %.2f' % (lime_fis, lime_rbs),
              'shap %.2f %.2f' % (shap_fis, shap_rbs),
              'maple %.2f %.2f' % (maple_fis, maple_rbs))

    df = pd.DataFrame(data=results)
    df = df[['black_box', 'n_records', 'n_all_features', 'n_features', 'n_coefficients', 'random_state',
             'idx', 'lime_cs', 'shap_cs', 'maple_cs', 'lime_f1', 'shap_f1', 'maple_f1']]
    # print(df.head())

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)


def main():

    n_records = 1000
    n_all_features_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_coefficients_list = [2, 4, 8, 16, 32, 64, 128, 256]
    exp_per_naf = 10
    path = '../results/'
    filename = path + 'tabular_linear_synthetic_black_box_type2.csv'
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
            n_features_list = [2]
        elif n_all_features == 4:
            n_features_list = [2, 3, 4]
        elif n_all_features == 8:
            n_features_list = [2, 3, 4, 5, 6, 7, 8]
        else:
            gap = n_all_features / exp_per_naf
            n_features_list = np.around(np.arange(2, n_all_features + 1, gap)).astype(int).tolist()
            n_features_list[-1] = n_all_features

        for n_features in n_features_list:
            if restart and n_all_features <= restart['n_all_features'] and n_features <= restart['n_features']:
                continue

            for n_coefficients in n_coefficients_list:
                if restart and n_all_features <= restart['n_all_features'] and n_features <= restart['n_features'] and \
                    n_coefficients < restart['n_coefficients']:
                    continue

                max_comb = np.max([scipy.special.comb(n_features, v) for v in range(n_features)])
                if n_coefficients > max_comb:
                    break

                flag = True
                attempts = 0
                while flag and attempts < max_attempts:
                    try:
                        print(datetime.datetime.now(), 'syege - tlsb2', 'black_box %s' % black_box,
                              'n_all_features %s' % n_all_features, 'n_features %s' % n_features,
                              'n_coefficients %s' % n_coefficients, 'rs %s' % random_state)
                        run(black_box, n_records, n_all_features, n_features, n_coefficients, random_state, filename)
                        flag = False
                    except ValueError:
                        attempts += 1
                    random_state += 1
                black_box += 1


if __name__ == "__main__":
    main()

