import os
import sys
sys.path.append('../lime')
sys.path.append('../maple')
sys.path.append('../syege')

import datetime
import numpy as np
import pandas as pd

from MAPLE import MAPLE
from shap import KernelExplainer
from lime_image import LimeImageExplainer
from scikit_image import SegmentationAlgorithm

from isyege import generate_synthetic_image_classifier
from isyege import generate_random_img_dataset
from isyege import get_pixel_importance_explanation
from evaluation import pixel_based_similarity


def run(black_box, n_records, img_size, cell_size, n_features, p_border, colors_p, random_state, filename):

    sic = generate_synthetic_image_classifier(img_size=img_size, cell_size=cell_size, n_features=n_features,
                                              p_border=p_border, random_state=random_state)

    pattern = sic['pattern']
    predict = sic['predict']
    predict_proba = sic['predict_proba']

    X_test = generate_random_img_dataset(pattern, nbr_images=n_records, pattern_ratio=0.5, img_size=img_size,
                                         cell_size=cell_size, min_nbr_cells=0.1, max_nbr_cells=0.3, colors_p=colors_p)

    Y_test_proba = predict_proba(X_test)
    Y_test = predict(X_test)

    lime_explainer = LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=10, ratio=0.5)
    tot_num_features = img_size[0] * img_size[1]

    background = np.array([np.zeros(img_size).ravel()] * 10)
    shap_explainer = KernelExplainer(predict_proba, background)

    nbr_records_explainer = 10
    idx_records_train_expl = np.random.choice(range(len(X_test)), size=nbr_records_explainer, replace=False)
    idx_records_test_expl = np.random.choice(range(len(X_test)), size=nbr_records_explainer, replace=False)

    Xm_train = np.array([x.ravel() for x in X_test[idx_records_train_expl]])
    Xm_test = np.array([x.ravel() for x in X_test[idx_records_test_expl]])

    print(datetime.datetime.now(), 'build maple')
    maple_explainer = MAPLE(Xm_train, Y_test_proba[idx_records_train_expl][:, 1],
                            Xm_test, Y_test_proba[idx_records_test_expl][:, 1],
                            n_estimators=100, max_features=0.5, min_samples_leaf=2)
    print(datetime.datetime.now(), 'build maple done')

    idx = 0
    results = list()
    for x, y in zip(X_test, Y_test):
        print(datetime.datetime.now(), 'seneca - text', 'black_box %s' % black_box,
              'n_features %s' % str(n_features), 'rs %s' % random_state, '%s/%s' % (idx, n_records), end=' ')

        gt_val = get_pixel_importance_explanation(x, sic)

        lime_exp = lime_explainer.explain_instance(x, predict_proba, top_labels=2, hide_color=0,
                                                   num_samples=10000, segmentation_fn=segmenter)
        _, lime_expl_val = lime_exp.get_image_and_mask(y, positive_only=True, num_features=tot_num_features,
                                                       hide_rest=False, min_weight=0.0)

        shap_expl_val = shap_explainer.shap_values(x.ravel(), l1_reg='bic')[1]
        shap_expl_val = np.sum(np.reshape(shap_expl_val, img_size), axis=2)
        tmp = np.zeros(shap_expl_val.shape)
        tmp[np.where(shap_expl_val > 0.0)] = 1.0
        shap_expl_val = tmp

        maple_exp = maple_explainer.explain(x)
        maple_expl_val = maple_exp['coefs'][:-1]
        maple_expl_val = np.sum(np.reshape(maple_expl_val, img_size), axis=2)
        tmp = np.zeros(maple_expl_val.shape)
        tmp[np.where(maple_expl_val > 0.0)] = 1.0
        maple_expl_val = tmp

        lime_f1, lime_pre, lime_rec = pixel_based_similarity(lime_expl_val.ravel(), gt_val, ret_pre_rec=True)
        shap_f1, shap_pre, shap_rec = pixel_based_similarity(shap_expl_val.ravel(), gt_val, ret_pre_rec=True)
        maple_f1, maple_pre, maple_rec = pixel_based_similarity(maple_expl_val.ravel(), gt_val, ret_pre_rec=True)

        res = {
            'black_box': black_box,
            'n_records': n_records,
            'img_size': '"%s"' % str(img_size),
            'cell_size': '"%s"' % str(cell_size),
            'n_features': '"%s"' % str(n_features),
            'random_state': random_state,
            'idx': idx,
            'lime_f1': lime_f1,
            'lime_pre': lime_pre,
            'lime_rec': lime_rec,
            'shap_f1': shap_f1,
            'shap_pre': shap_pre,
            'shap_rec': shap_rec,
            'maple_f1': maple_f1,
            'maple_pre': maple_pre,
            'maple_rec': maple_rec,
        }
        results.append(res)
        print('lime %.2f' % lime_f1, 'shap %.2f' % shap_f1, 'maple %.2f' % maple_f1)

        idx += 1

    df = pd.DataFrame(data=results)
    df = df[['black_box', 'n_records', 'img_size', 'cell_size', 'n_features', 'random_state', 'idx',
             'lime_f1', 'lime_pre', 'lime_rec', 'shap_f1', 'shap_pre', 'shap_rec', 'maple_f1', 'maple_pre', 'maple_rec',
             ]]
    # print(df.head())

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)


def main():

    n_records = 1000
    n_features_list = [(8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (32, 32)]
    nbr_test_per_feature = 10
    p_border_list = [0.0, 0.25, 0.5, 0.75, 1.0]

    img_size = (32, 32, 3)
    cell_size = (4, 4)
    colors_p = np.array([0.15, 0.7, 0.15])

    path = '../results/'
    filename = path + 'image_synthetic_black_box_new.csv'

    restart = None
    if os.path.isfile(filename):
        restart = pd.read_csv(filename).tail(1).to_dict('record')[0]
        print('restart', restart)

    black_box = 0
    random_state = 0
    if restart:
        # black_box = restart['black_box'] + 1
        random_state = restart['random_state'] + 1
    for n_features in n_features_list:
        if restart and n_features < restart['n_features']:
            continue
        for p_border in p_border_list:
            if restart and n_features <= restart['n_features'] and p_border < restart['p_border']:
                continue

            if n_features[0] <= 12 and p_border > 0.0:
                continue

            for test_id in range(nbr_test_per_feature):

                if restart and n_features <= restart['n_features'] and black_box < restart['black_box']:
                    black_box += 1
                    continue

                print(datetime.datetime.now(), 'seneca - image', 'black_box %s' % black_box,
                      'n_features %s' % str(n_features), 'rs %s' % random_state)
                run(black_box, n_records, img_size, cell_size, n_features, p_border, colors_p, random_state, filename)

                random_state += 1
                black_box += 1


if __name__ == "__main__":
    main()
