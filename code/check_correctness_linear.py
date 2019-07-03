import numpy as np

from syege import generate_synthetic_linear_classifier
from syege import get_feature_importance_explanation


def main():

    m = 5
    n = 10

    n_features = 3
    random_state = 1

    num_operations = 10
    p_binary = 0.7
    p_parenthesis = 0.3

    slc = generate_synthetic_linear_classifier(expr='x0**3-2*x1**2+3*x2', n_features=n_features, n_all_features=m,
                                               random_state=random_state, num_operations=num_operations,
                                               p_binary=p_binary, p_parenthesis=p_parenthesis)

    expr = slc['expr']
    X = slc['X']
    Y = slc['Y']
    feature_names = slc['feature_names']

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))

    print(expr)

    def real_explanation(x):
        return np.array([3*x[0]**2, -4*x[1], 3.0])

    for x in X_test:
        expl_val, cx = get_feature_importance_explanation(x, slc, n_features, get_values=True, get_closest_point=True)
        print(x, cx, np.array_equal(expl_val[:n_features], real_explanation(cx)))
        print(expl_val)
        print(real_explanation(cx))
        print()


if __name__ == "__main__":
    main()
