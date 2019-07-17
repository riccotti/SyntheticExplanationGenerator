import numpy as np

from shap import KernelExplainer
from sklearn.datasets import fetch_20newsgroups

from tsyege import generate_synthetic_text_classifier, preprocess_data
from tsyege import get_word_importance_explanation
from evaluation import word_based_similarity


def main():
    n_features = 100
    random_state = 0

    X_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_train = preprocess_data(X_train)

    print('qui')
    stc = generate_synthetic_text_classifier(X_train, n_features=n_features, random_state=random_state)

    predict = stc['predict']
    predict_proba = stc['predict_proba']
    words = stc['words_vec']
    vectorizer = stc['vectorizer']
    nbr_terms = stc['nbr_terms']

    X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None).data
    X_test = preprocess_data(X_test)
    X_test = vectorizer.transform(X_test).toarray()
    print('quo')

    # reference = np.zeros(nbr_terms)
    # explainer = KernelExplainer(predict_proba, np.reshape(reference, (1, len(reference))))

    sentences_length = list()
    for x in X_test:
        sentences_length.append(len(np.where(x != 0)[0]))

    print('qua')
    avg_nbr_words = np.mean(sentences_length)
    std_nbr_words = np.std(sentences_length)
    words_with_weight = np.where(words != 0)[0]
    print(avg_nbr_words, std_nbr_words)
    print(words_with_weight)
    reference = list()
    for i in range(10):
        nbr_words_in_sentence = int(np.random.normal(avg_nbr_words, std_nbr_words))
        selected_words = np.random.choice(range(nbr_terms), size=nbr_words_in_sentence, replace=False)
        print(i, nbr_words_in_sentence, len(set(selected_words) & set(words_with_weight)))
        while len(set(selected_words) & set(words_with_weight)) > 0:
            nbr_words_in_sentence = int(np.random.normal(avg_nbr_words, std_nbr_words))
            selected_words = np.random.choice(range(nbr_terms), size=nbr_words_in_sentence, replace=False)
            print(i, nbr_words_in_sentence, len(set(selected_words) & set(words_with_weight)))
        sentence = np.zeros(nbr_terms)
        sentence[selected_words] = 1.0
        reference.append(sentence)
        print('')
    reference = np.array(reference)
    print(reference)
    explainer = KernelExplainer(predict_proba, reference) #X_test[:10])

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

