from sklearn.datasets import fetch_20newsgroups
from lime_text import LimeTextExplainer

from tsyege import generate_synthetic_text_classifier, preprocess_data
from tsyege import get_word_importance_explanation
from evaluation import word_based_similarity


import numpy as np

from shap import KernelExplainer


from syege import generate_synthetic_linear_classifier
from syege import get_feature_importance_explanation
from evaluation import feature_importance_similarity


def main():
    n_features = 100
    random_state = 0

    stc = generate_synthetic_text_classifier(n_features=n_features, random_state=random_state)

    predict = stc['predict']
    predict_proba = stc['predict_proba']
    words = stc['words']
    vectorizer = stc['vectorizer']
    nbr_terms = stc['nbr_terms']

    X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_test = preprocess_data(X_test)
    X_test = vectorizer.transform(X_test).toarray()

    reference = np.zeros(nbr_terms)
    # explainer = KernelExplainer(predict_proba, np.reshape(reference, (1, len(reference))))
    explainer = KernelExplainer(predict_proba, X_test[:10])

    for x in X_test[:10]:
        expl_val = explainer.shap_values(x, l1_reg='bic')[1]
        gt_val = get_word_importance_explanation(x, stc)
        wbs = word_based_similarity(expl_val, gt_val, use_values=False)
        # print(expl_val)
        # print(gt_val)
        print(wbs, word_based_similarity(expl_val, gt_val, use_values=True))
        print('')


if __name__ == "__main__":
    main()

