import os
import sys
sys.path.append('../lime')
sys.path.append('../anchor')
sys.path.append('../lore')
sys.path.append('../rulematrix')
sys.path.append('../syege')

import datetime
import numpy as np
import pandas as pd

from lorem import LOREM
from lore_datamanager import prepare_dataset
from lore_util import neuclidean
from rulematrix import Surrogate
from anchor_tabular import AnchorTabularExplainer

from syege import get_rule_explanation_complete
from syege import generate_syntetic_rule_based_classifier
from evaluation import rule_based_similarity_complete


sampling_map = {
    2: 0.5,
    3: 0.5,
    4: 0.5,
    5: 0.5,
    6: 0.5,
    7: 0.5,
    8: 1.0,
    9: 1.0,
    10: 2.0,
}


def run(black_box, n_records, n_all_features, n_features, random_state, filename):

    n = n_records
    m = n_all_features

    factor = 10
    # sampling = 0.5
    if n_features in sampling_map:
        sampling = sampling_map[n_features]
    else:
        sampling = n_features / 10

    srbc = generate_syntetic_rule_based_classifier(n_samples=n_records, n_features=n_features, n_all_features=m,
                                                   random_state=random_state, factor=factor, sampling=sampling,
                                                   explore_domain=False)

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

    anchor_explainer = AnchorTabularExplainer(class_names=class_values, feature_names=feature_names,
                                              categorical_names={})
    anchor_explainer.fit(X_test, Y_test, X_test, Y_test)

    lore_explainer = LOREM(X_test, predict, feature_names, class_name, class_values, numeric_columns, features_map,
                           neigh_type='geneticp', categorical_use_prob=True, bb_predict_proba=predict_proba,
                           continuous_fun_estimation=True, size=100, ocr=0.1, multi_label=False, one_vs_rest=False,
                           filter_crules=True, random_state=0, verbose=False, ngen=10)

    sbrl_explainer = Surrogate(predict, student=None, is_continuous=None, is_categorical=None, is_integer=None,
                               ranges=None, cov_factor=1.0, sampling_rate=2.0, seed=None, verbose=False)
    if n_features <= 5:
        sbrl_explainer.fit(X)

    results = list()
    for idx, x in enumerate(X_test):
        print(datetime.datetime.now(), 'syege - trsb', 'black_box %s' % black_box,
              'n_all_features %s' % n_all_features, 'n_features %s' % n_features, 'rs %s' % random_state,
              '%s/%s' % (idx, n_records), end=' ')
        gt_dict = get_rule_explanation_complete(x, srbc, n_features)

        anchor_flag = True
        try:
            anchor_exp, anchor_exp_dict = anchor_explainer.explain_instance_ric(x, predict, threshold=0.95)
            anchor_expl_val = np.array([1 if e in anchor_exp.features() else 0 for e in range(m)])

        except MemoryError:
            print(datetime.datetime.now(), 'memory error anchor')
            anchor_flag = False

        lore_flag = True
        while lore_flag:
            try:
                lore_exp = lore_explainer.explain_instance(x, samples=1000, use_weights=True, metric=neuclidean)
                lore_flag = False
            except:
                print(datetime.datetime.now(), 'retry lore')
                lore_flag = True

        lore_expl_dict = dict()
        for f in feature_names:
            lore_expl_dict[(f, '<=')] = np.inf
            lore_expl_dict[(f, '>')] = -np.inf

        for p in lore_exp.rule.premises:
            lore_expl_dict[(p.att, p.op)] = p.thr

        if n_features <= 5:
            sbrl_expl_val, sbrl_exp_dict = sbrl_explainer.explain_ric(x, m)

        if anchor_flag:
            anchor_rbs = rule_based_similarity_complete(anchor_exp_dict, gt_dict, eps=0.1)
        else:
            anchor_rbs = -1.0

        lore_rbs = rule_based_similarity_complete(lore_expl_dict, gt_dict, eps=0.1)

        if n_features <= 5:
            sbrl_rbs = rule_based_similarity_complete(sbrl_exp_dict, gt_dict)
        else:
            sbrl_rbs = -1.0

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
            'anchor': anchor_rbs,
            'lore': lore_rbs,
            'sbrl': sbrl_rbs,
        }
        results.append(res)
        print('anchor %s' % anchor_rbs, 'lore %s' % lore_rbs, 'sbrl %s ' % sbrl_rbs)

        if idx > 0 and idx % 10 == 0:
            df = pd.DataFrame(data=results)
            df = df[['black_box', 'n_records', 'n_all_features', 'n_features', 'random_state',
                     'idx', 'anchor', 'lore', 'sbrl']]
            # print(df.head())

            if not os.path.isfile(filename):
                df.to_csv(filename, index=False)
            else:
                df.to_csv(filename, mode='a', index=False, header=False)
            results = list()

    df = pd.DataFrame(data=results)
    df = df[['black_box', 'n_records', 'n_all_features', 'n_features', 'random_state',
             'idx', 'anchor', 'lore', 'sbrl']]
    # print(df.head())

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)


def main():

    n_records = 1000
    # n_all_features_list = [int(sys.argv[1])]  #[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_all_features = int(sys.argv[1])
    exp_per_naf = 10
    path = '../results/'
    filename = path + 'tabular_rulebased_synthetic_black_box_new_%s_complete.csv' % n_all_features
    random_state = 0
    max_attempts = 100

    restart = None
    if os.path.isfile(filename):
        restart = pd.read_csv(filename).tail(1).to_dict('record')[0]
        print('restart', restart)

    # black_box = n_all_features_list[0]
    black_box = n_all_features
    if restart:
        black_box = restart['black_box'] + 1
        random_state = restart['random_state'] + 1
    for n_all_features in [n_all_features]:
        if restart and n_all_features < restart['n_all_features']:
            continue
        # if n_all_features == 2:
        #     n_features_list = [2] * exp_per_naf
        # elif n_all_features == 4:
        #     n_features_list = [2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        # elif n_all_features == 8:
        #     n_features_list = [2, 2, 3, 3, 4, 4, 5, 6, 7, 8]
        # else:
        #     gap = n_all_features / exp_per_naf
        #     n_features_list_a = np.around(np.arange(2, n_all_features + 1, gap)).astype(int).tolist()
        #     # n_features_list[-1] = n_all_features
        #     n_features_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        #     n_features_list = [x for x in n_features_list if x <= n_all_features]
        #     n_features_list.extend(n_features_list_a)
        #     n_features_list = sorted(n_features_list)
        n_features_list = [2, 4, 8, 16, 32]

        for n_features in n_features_list:
            if restart and n_all_features <= restart['n_all_features'] and n_features <= restart['n_features']:
                continue

            flag = True
            attempts = 0
            while flag and attempts < max_attempts:
                # try:
                print(datetime.datetime.now(), 'syege - trsb', 'black_box %s' % black_box,
                      'n_all_features %s' % n_all_features, 'n_features %s' % n_features, 'rs %s' % random_state)
                run(black_box, n_records, n_all_features, n_features, random_state, filename)
                flag = False
                # except ValueError:
                #     attempts += 1
                random_state += 1
            black_box += 1


if __name__ == "__main__":
    main()

