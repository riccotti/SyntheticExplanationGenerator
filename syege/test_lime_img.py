from sklearn.datasets import fetch_20newsgroups
from lime_image import LimeImageExplainer

from tsyege import generate_synthetic_text_classifier, preprocess_data
from tsyege import get_word_importance_explanation
from evaluation import word_based_similarity


def main():
    #  TODO

    n_features = 100

    stc = generate_synthetic_text_classifier(n_features=n_features)

    predict = stc['predict']
    predict_proba = stc['predict_proba']
    words = stc['words']

    # print(words)

    X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_test = preprocess_data(X_test)


    explainer = LimeImageExplainer()

    for x in X_test[:10]:
        # print(x)
        exp = explainer.explain_instance(x, predict_proba, num_features=n_features)
        expl_val = {e[0]: e[1] for e in exp.as_list()}
        gt_val = get_word_importance_explanation(x, stc)
        wbs = word_based_similarity(expl_val, gt_val, use_values=False)
        print(expl_val)
        print(gt_val)
        print(wbs, word_based_similarity(expl_val, gt_val, use_values=True))
        print('')


if __name__ == "__main__":
    main()

