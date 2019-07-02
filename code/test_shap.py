import numpy as np

from shap import KernelExplainer


from syege import generate_synthetic_linear_classifier
from syege import get_feature_importance_explanation
from evaluation import feature_importance_similarity


def main():
    m = 5
    n = 10

    n_features = 2
    random_state = 1

    num_operations = 10
    p_binary = 0.7
    p_parenthesis = 0.3

    slc = generate_synthetic_linear_classifier(expr='x0-x1*sin(x1)**2', n_features=n_features, n_all_features=m,
                                               random_state=random_state, num_operations=num_operations,
                                               p_binary=p_binary, p_parenthesis=p_parenthesis)
    expr = slc['expr']
    X = slc['X']
    if slc['feature_names'] is None:
        slc['feature_names'] = ['x%s' % i for i in range(m)]
    predict_proba = slc['predict_proba']

    print(expr)

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))

    reference = np.zeros(m)
    explainer = KernelExplainer(predict_proba, np.reshape(reference, (1, len(reference))))

    for i, x in enumerate(X_test):
        print(x)
        expl_val = explainer.shap_values(x)[1]
        gt_val = get_feature_importance_explanation(x, slc, n_features, get_values=True)
        fis = feature_importance_similarity(expl_val, gt_val)
        print(expl_val)
        print(gt_val)
        print(fis)
        print('')
        if i == 10:
            break


if __name__ == "__main__":
    main()

