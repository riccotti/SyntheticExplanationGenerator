from lime_text import LimeTextExplainer
from sklearn.datasets import fetch_20newsgroups

from tsyege import generate_synthetic_text_classifier, preprocess_data
from tsyege import get_word_importance_explanation_text
from evaluation import word_based_similarity_text


def main():
    n_features = 100
    random_state = 0

    X_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_train = preprocess_data(X_train)

    stc = generate_synthetic_text_classifier(X_train, n_features=n_features, use_textual_words=True,
                                             random_state=random_state)

    predict = stc['predict']
    predict_proba = stc['predict_proba']
    words = stc['words']
    vectorizer = stc['vectorizer']
    nbr_terms = stc['nbr_terms']

    # print(words)

    X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_test = preprocess_data(X_test)

    explainer = LimeTextExplainer(class_names=[0, 1])

    for x in X_test[:10]:
        # print(x)
        exp = explainer.explain_instance(x, predict_proba, num_features=n_features)
        expl_val = {e[0]: e[1] for e in exp.as_list()}

        gt_val = get_word_importance_explanation_text(x, stc)
        print(gt_val)
        print(expl_val)
        wbs = word_based_similarity_text(expl_val, gt_val, use_values=False)
        print(wbs, word_based_similarity_text(expl_val, gt_val, use_values=True))
        print('')


if __name__ == "__main__":
    main()

