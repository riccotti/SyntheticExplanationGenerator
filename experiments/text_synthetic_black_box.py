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
from lime_text import LimeTextExplainer
from sklearn.datasets import fetch_20newsgroups

from tsyege import generate_synthetic_text_classifier, preprocess_data
from tsyege import get_word_importance_explanation_text, get_word_importance_explanation
from evaluation import word_based_similarity, word_based_similarity_text


def get_reference4shap(X, words, nbr_terms, nbr_references=10):
    sentences_length = list()
    for x in X:
        sentences_length.append(len(np.where(x != 0)[0]))

    avg_nbr_words = np.mean(sentences_length)
    std_nbr_words = np.std(sentences_length)
    min_nbr_words = np.min(sentences_length)
    max_nbr_words = np.max(sentences_length)
    words_with_weight = np.where(words != 0)[0]

    reference = list()
    for i in range(nbr_references):
        nbr_words_in_sentence = int(np.random.normal(avg_nbr_words, std_nbr_words))
        nbr_words_in_sentence = min(max(nbr_words_in_sentence, min_nbr_words), max_nbr_words)
        selected_words = np.random.choice(range(nbr_terms), size=nbr_words_in_sentence, replace=False)
        while len(set(selected_words) & set(words_with_weight)) > 0:
            nbr_words_in_sentence = min(max(nbr_words_in_sentence, min_nbr_words), max_nbr_words)
            selected_words = np.random.choice(range(nbr_terms), size=nbr_words_in_sentence, replace=False)

        sentence = np.zeros(nbr_terms)
        sentence[selected_words] = 1.0
        reference.append(sentence)

    reference = np.array(reference)
    return reference


def run(black_box, n_records, n_features, random_state, filename):

    X_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_train = preprocess_data(X_train)

    stc = generate_synthetic_text_classifier(X_train, n_features=n_features, random_state=random_state)

    predict_proba = stc['predict_proba']
    vectorizer = stc['vectorizer']
    nbr_terms = stc['nbr_terms']

    X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_test = preprocess_data(X_test)
    X_test_nbrs = vectorizer.transform(X_test).toarray()
    Y_test = predict_proba(X_test_nbrs)

    lime_explainer = LimeTextExplainer(class_names=[0, 1])

    reference = get_reference4shap(X_test_nbrs, stc['words_vec'], nbr_terms, nbr_references=10)
    shap_explainer = KernelExplainer(predict_proba, reference)

    # print(idx_records_train_expl)
    # print(X_test_nbrs[idx_records_train_expl])
    # print(Y_test[idx_records_train_expl][:, 1])
    # print(np.any(np.isnan(X_test_nbrs[idx_records_train_expl])))
    # print(np.any(np.isnan(X_test_nbrs[idx_records_test_expl])))
    # print(np.any(np.isnan(Y_test[idx_records_train_expl][:, 1])))
    # print(np.any(np.isnan(Y_test[idx_records_test_expl][:, 1])))

    nbr_records_explainer = 100
    idx_records_train_expl = np.random.choice(range(len(X_test)), size=nbr_records_explainer, replace=False)
    idx_records_test_expl = np.random.choice(range(len(X_test)), size=nbr_records_explainer, replace=False)

    # print(datetime.datetime.now(), 'build maple')
    maple_explainer = MAPLE(X_test_nbrs[idx_records_train_expl], Y_test[idx_records_train_expl][:, 1],
                            X_test_nbrs[idx_records_test_expl], Y_test[idx_records_test_expl][:, 1],
                            n_estimators=100, max_features=0.5, min_samples_leaf=2)
    # print(datetime.datetime.now(), 'build maple done')

    results = list()
    explained = 0
    for idx, x in enumerate(X_test):
        x_nbrs = X_test_nbrs[idx]

        print(datetime.datetime.now(), 'seneca - text', 'black_box %s' % black_box,
              'n_features %s' % n_features, 'rs %s' % random_state, '%s/%s' % (idx, n_records), end=' ')

        gt_val_text = get_word_importance_explanation_text(x, stc)
        gt_val = get_word_importance_explanation(x_nbrs, stc)

        try:
            lime_exp = lime_explainer.explain_instance(x, predict_proba, num_features=n_features)
            lime_expl_val = {e[0]: e[1] for e in lime_exp.as_list()}

            shap_expl_val = shap_explainer.shap_values(x_nbrs, l1_reg='bic')[1]

            maple_exp = maple_explainer.explain(x_nbrs)
            maple_expl_val = maple_exp['coefs'][:-1]
        except ValueError:
            print(datetime.datetime.now(), 'Error in explanation')
            continue

        lime_cs = word_based_similarity_text(lime_expl_val, gt_val_text, use_values=True)
        lime_f1, lime_pre, lime_rec = word_based_similarity_text(lime_expl_val, gt_val_text,
                                                                 use_values=False, ret_pre_rec=True)

        shap_cs = word_based_similarity(shap_expl_val, gt_val, use_values=True)
        shap_f1, shap_pre, shap_rec = word_based_similarity(shap_expl_val, gt_val,
                                                            use_values=False, ret_pre_rec=True)

        maple_cs = word_based_similarity(maple_expl_val, gt_val, use_values=True)
        maple_f1, maple_pre, maple_rec = word_based_similarity(maple_expl_val, gt_val,
                                                               use_values=False, ret_pre_rec=True)

        # print(gt_val)
        # print(lime_expl_val)
        # print(shap_expl_val)
        # print(maple_expl_val)

        res = {
            'black_box': black_box,
            'n_records': n_records,
            'nbr_terms': nbr_terms,
            'n_features': n_features,
            'random_state': random_state,
            'idx': idx,
            'lime_cs': lime_cs,
            'lime_f1': lime_f1,
            'lime_pre': lime_pre,
            'lime_rec': lime_rec,
            'shap_cs': shap_cs,
            'shap_f1': shap_f1,
            'shap_pre': shap_pre,
            'shap_rec': shap_rec,
            'maple_cs': maple_cs,
            'maple_f1': maple_f1,
            'maple_pre': maple_pre,
            'maple_rec': maple_rec,
        }
        results.append(res)
        print('lime %.2f %.2f' % (lime_cs, lime_f1),
              'shap %.2f %.2f' % (shap_cs, shap_f1),
              'maple %.2f %.2f' % (maple_cs, maple_f1))

        explained += 1
        if explained >= n_records:
            break

    df = pd.DataFrame(data=results)
    df = df[['black_box', 'n_records', 'nbr_terms', 'n_features', 'random_state', 'idx',
             'lime_cs', 'lime_f1', 'lime_pre', 'lime_rec',
             'shap_cs', 'shap_f1', 'shap_pre', 'shap_rec',
             'maple_cs', 'maple_f1', 'maple_pre', 'maple_rec',
             ]]
    # print(df.head())

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)


def main():

    n_records = 1000
    n_features_list = [10, 25, 50, 100, 250, 500, 1000]
    nbr_test_per_feature = 10
    path = '../results/'
    filename = path + 'text_synthetic_black_box_new.csv'
    random_state = 0

    restart = None
    if os.path.isfile(filename):
        restart = pd.read_csv(filename).tail(1).to_dict('record')[0]
        print('restart', restart)

    black_box = 0
    if restart:
        # black_box = restart['black_box'] + 1
        random_state = restart['random_state'] + 1
    for n_features in n_features_list:
        if restart and n_features < restart['n_features']:
            continue

        for test_id in range(nbr_test_per_feature):

            if restart and n_features <= restart['n_features'] and black_box < restart['black_box']:
                black_box += 1
                continue

            print(datetime.datetime.now(), 'seneca - text', 'black_box %s' % black_box,
                  'n_features %s' % n_features, 'rs %s' % random_state)
            run(black_box, n_records, n_features, random_state, filename)

            random_state += 1
            black_box += 1


if __name__ == "__main__":
    main()

