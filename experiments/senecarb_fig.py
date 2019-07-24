import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from rule import get_rules
from syege import generate_syntetic_rule_based_classifier


def main():
    m = 2
    n = 1000

    n_features = 2
    random_state = 1

    factor = 2
    sampling = 0.15

    srbc = generate_syntetic_rule_based_classifier(n_features=n_features, n_all_features=m, random_state=random_state,
                                                   factor=factor, sampling=sampling, explore_domain=True)

    X = srbc['X']
    Y = srbc['Y']
    Ypp = srbc['Ypp']
    X0 = StandardScaler().fit_transform(srbc['X0'])
    Y0 = srbc['Y0']


    feature_names = srbc['feature_names']
    class_values = srbc['class_values']
    class_name = srbc['class_name']
    predict_proba = srbc['predict_proba']
    predict = srbc['predict']
    ff = srbc['ff']
    dt = srbc['dt']

    X_test = np.random.uniform(np.min(X), np.max(X), size=(n, m))


    plt.figure(figsize=(8, 8))
    cm = plt.cm.PRGn
    plt.scatter(X0[:, 0], X0[:, 1], c=Y0, s=5, cmap=cm)
    plt.xlim(X0[:, 0].min(), X0[:, 0].max())
    plt.ylim(X0[:, 1].min(), X0[:, 1].max())
    # plt.xticks(())
    # plt.yticks(())
    plt.ylabel(r'$X_1$', fontsize=24)
    plt.xlabel(r'$X_0$', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.savefig('../fig/make_classification.png', format='png', bbox_inches='tight')
    plt.show()

    dt0 = DecisionTreeClassifier()
    dt0.fit(X0, Y0)

    f_min = [X0[:, i].min() - sampling * 2 for i in range(n_features)]
    f_max = [X0[:, i].max() + sampling * 2 for i in range(n_features)]

    ff0 = np.meshgrid(*[np.arange(f_min[i], f_max[i], sampling) for i in range(n_features)], copy=False)
    values = [ff0[i].ravel() for i in range(n_features)]
    X_new0 = np.c_[values].T
    Y_new0 = dt0.predict(X_new0)

    plt.figure(figsize=(8, 8))
    cm = plt.cm.PRGn
    plt.contourf(ff0[0], ff0[1], Y_new0.reshape(ff0[0].shape), cmap=cm, alpha=0.3)
    plt.scatter(X_new0[:, 0], X_new0[:, 1], c=Y_new0, s=5, cmap=cm)
    plt.xlim(X_new0[:, 0].min(), X_new0[:, 0].max())
    plt.ylim(X_new0[:, 1].min(), X_new0[:, 1].max())
    # plt.xticks(())
    # plt.yticks(())
    plt.ylabel(r'$X_1$', fontsize=24)
    plt.xlabel(r'$X_0$', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.savefig('../fig/train_boundaries_dt0.png', format='png', bbox_inches='tight')
    plt.show()

    Y_test = dt0.predict(X_test[:, :n_features])
    plt.figure(figsize=(8, 8))
    cm = plt.cm.PRGn
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=5, cmap=cm)
    plt.contourf(ff0[0], ff0[1], Y_new0.reshape(ff0[0].shape), cmap=cm, alpha=0.3)
    plt.xlim(ff0[0].min(), ff0[0].max())
    plt.ylim(ff0[1].min(), ff0[1].max())
    # plt.xticks(())
    # plt.yticks(())
    plt.ylabel(r'$X_1$', fontsize=24)
    plt.xlabel(r'$X_0$', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.savefig('../fig/test_dt0.png', format='png', bbox_inches='tight')
    plt.show()

    # plt.figure(figsize=(8, 8))
    # cm = plt.cm.PRGn
    # plt.contourf(ff[0], ff[1], Y.reshape(ff[0].shape), cmap=cm, alpha=1)
    # plt.xlim(ff[0].min(), ff[0].max())
    # plt.ylim(ff[1].min(), ff[1].max())
    # # plt.xticks(())
    # # plt.yticks(())
    # plt.show()
    #
    # plt.figure(figsize=(8, 8))
    # cm = plt.cm.PRGn
    # plt.scatter(X[:, 0], X[:, 1], c=Y, s=5, cmap=cm)
    # plt.xlim(X[:, 0].min(), X[:, 0].max())
    # plt.ylim(X[:, 1].min(), X[:, 1].max())
    # # plt.xticks(())
    # # plt.yticks(())
    # plt.show()

    plt.figure(figsize=(8, 8))
    cm = plt.cm.PRGn
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=5, cmap=cm)
    plt.contourf(ff[0], ff[1], Y.reshape(ff[0].shape), cmap=cm, alpha=0.3)
    plt.xlim(ff[0].min(), ff[0].max())
    plt.ylim(ff[1].min(), ff[1].max())
    # plt.xticks(())
    # plt.yticks(())
    plt.ylabel(r'$X_1$', fontsize=24)
    plt.xlabel(r'$X_0$', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.savefig('../fig/train_boundaries_dt.png', format='png', bbox_inches='tight')
    plt.show()

    Y_test = dt.predict(X_test[:, :n_features])
    plt.figure(figsize=(8, 8))
    cm = plt.cm.PRGn
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=5, cmap=cm)
    plt.contourf(ff[0], ff[1], Y.reshape(ff[0].shape), cmap=cm, alpha=0.3)
    plt.xlim(ff[0].min(), ff[0].max())
    plt.ylim(ff[1].min(), ff[1].max())
    # plt.xticks(())
    # plt.yticks(())
    plt.ylabel(r'$X_1$', fontsize=24)
    plt.xlabel(r'$X_0$', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.savefig('../fig/test_dt.png', format='png', bbox_inches='tight')
    plt.show()

    rules0 = get_rules(dt0, feature_names, class_name, class_values, feature_names)
    rules = get_rules(dt, feature_names, class_name, class_values, feature_names)

    print('dt0', len(rules0))
    print('dt', len(rules))


if __name__ == "__main__":
    main()

