import numpy as np
import rulematrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
data = load_iris()
X, y = data['data'], data['target']


teacher = DecisionTreeClassifier()
teacher.fit(X, y)

# surrogate = rulematrix.Surrogate(teacher.predict, student=None, is_continuous=None, is_categorical=None, is_integer=None,
#                                  ranges=None, cov_factor=1.0, sampling_rate=2.0, seed=None,
#                                  rlargs={'feature_names': feature_names, 'verbose': 2}, verbose=False)
# surrogate.fit(X)

feature_names = ['sl', 'sw', 'pl', 'pw']
explainer = rulematrix.rule_surrogate(teacher.predict, X, is_continuous=None, is_categorical=None, is_integer=None,
                   ranges=None, cov_factor=1.0, sampling_rate=2.0, seed=0,
                                      rlargs={'feature_names': feature_names, 'verbose': 2})


rl = explainer.student
print(rl)

# print(rl.n_rules)
print(rl._rule_list)
print('-----')

dp = rl.decision_path(X[70].reshape(1,-1))
print(dp)
# print(surrogate.student.explain(X[58]))
# print(surrogate.student.predict_proba(X))
features_used = np.zeros(4)
for r, ru in zip(rl._rule_list, dp):
    if ru:
        for c in r[0]:
            features_used[c.feature_idx] = 1

print(features_used)

print(np.array([]))

