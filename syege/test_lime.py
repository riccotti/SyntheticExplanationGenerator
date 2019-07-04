import numpy as np

from lime.lime_tabular import LimeTabularExplainer


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
    feature_names = slc['feature_names']
    class_values = slc['class_values']
    predict_proba = slc['predict_proba']

    print(expr)

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))

    explainer = LimeTabularExplainer(X_test, feature_names=feature_names, class_names=class_values,
                                     discretize_continuous=False, discretizer='entropy')

    for x in X_test:
        print(x)
        exp = explainer.explain_instance(x, predict_proba, num_features=m)
        expl_val = np.array([e[1] for e in exp.as_list()])
        gt_val = get_feature_importance_explanation(x, slc, n_features, get_values=True)
        fis = feature_importance_similarity(expl_val, gt_val)
        print(expl_val)
        print(gt_val)
        print(fis)
        print('')
        break


if __name__ == "__main__":
    main()

