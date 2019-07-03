import numpy as np

from anchor.anchor_tabular import AnchorTabularExplainer


from syege import get_rule_explanation
from syege import generate_syntetic_rule_based_classifier
from evaluation import rule_based_similarity


def main():
    from sklearn.datasets import load_iris
    from skrules import SkopeRules

    dataset = load_iris()
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    clf = SkopeRules(max_depth_duplication=2,
                     n_estimators=30,
                     precision_min=0.3,
                     recall_min=0.1,
                     feature_names=feature_names)

    # for idx, species in enumerate(dataset.target_names):
    #     X, y = dataset.data, dataset.target
    #     clf.fit(X, y == idx)
    #     rules = clf.rules_[0:3]
    #     print("Rules for iris", species)
    #     for rule in rules:
    #         print(rule)
    #     print()
    #     print(20 * '=')
    #     print()

    y_score = clf.score_top_rules(X)  # Get a risk score for each test example
    print(y_score)


if __name__ == "__main__":
    main()

