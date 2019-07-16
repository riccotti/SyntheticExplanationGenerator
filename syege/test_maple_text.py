import numpy as np

from MAPLE import MAPLE
from sklearn.datasets import fetch_20newsgroups

from tsyege import generate_synthetic_text_classifier, preprocess_data
from tsyege import get_word_importance_explanation, get_word_importance_explanation_text
from evaluation import word_based_similarity


# MAPLE fa schifo ma si puo' usareee
def main():
    n_features = 100
    random_state = 0

    stc = generate_synthetic_text_classifier(n_features=n_features, use_textual_words=False,
                                             random_state=random_state)

    predict = stc['predict']
    predict_proba = stc['predict_proba']
    words = stc['words']
    vectorizer = stc['vectorizer']
    nbr_terms = stc['nbr_terms']

    # print(words)

    X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                categories=None).data
    X_test = preprocess_data(X_test)
    X_test = vectorizer.transform(X_test).toarray()
    Y_test = predict(X_test)

    print('building')
    explainer = MAPLE(X_test[:10], Y_test[:10], X_test[:10], Y_test[:10])
    print('built')

    for x in X_test[:10]:
        # print(x)
        exp = explainer.explain(x)
        expl_val = exp['coefs'][:-1]
        gt_val = get_word_importance_explanation(x, stc)
        # gt_val = get_word_importance_explanation_text(x, stc)
        # print(gt_val)
        # print(expl_val)
        # print(np.where(gt_val != 0))
        # print(np.where(expl_val != 0))

        wbs = word_based_similarity(expl_val, gt_val, use_values=False)
        print(wbs, word_based_similarity(expl_val, gt_val, use_values=True))
        print('')


if __name__ == "__main__":
    main()

