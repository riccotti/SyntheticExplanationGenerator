import spacy

from sklearn.datasets import fetch_20newsgroups
from anchor_text import AnchorText

from tsyege import generate_synthetic_text_classifier, preprocess_data
from tsyege import get_word_importance_explanation_text
from evaluation import word_based_similarity_text


def main():
    n_features = 100

    stc = generate_synthetic_text_classifier(n_features=n_features, use_textual_words=True)

    predict = stc['predict']
    predict_proba = stc['predict_proba']
    words = stc['words']
    vectorizer = stc['vectorizer']
    nbr_terms = stc['nbr_terms']

    # print(words)

    X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                categories=None).data
    X_test = preprocess_data(X_test)

    nlp = spacy.load('en')
    explainer = AnchorText(nlp, class_names=[0, 1], use_unk_distribution=True)

    for x in X_test[:10]:
        # print(x)
        exp = explainer.explain_instance(str(x), predict, threshold=0.95, use_proba=True)
        expl_val = {e[0]: 1.0 for e in exp.names()}

        gt_val = get_word_importance_explanation_text(x, stc)
        print(gt_val)
        print(expl_val)
        wbs = word_based_similarity_text(expl_val, gt_val, use_values=True)
        print(wbs)
        print('')


if __name__ == "__main__":
    main()

