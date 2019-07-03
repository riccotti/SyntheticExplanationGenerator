
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes

from RuleListClassifier import *

dataseturls = ["https://archive.ics.uci.edu/ml/datasets/Iris", "https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes"]
datasets = ["iris", "diabetes"]
data_feature_labels = [
    ["Sepal length", "Sepal width", "Petal length", "Petal width"],
    ["#Pregnant","Glucose concentration demo","Blood pressure(mmHg)",
     "Triceps skin fold thickness(mm)","2-Hour serum insulin (mu U/ml)",
     "Body mass index","Diabetes pedigree function","Age (years)", "Pippo", "Pluto"]
]
data_class1_labels = ["Iris Versicolour", "No Diabetes"]

for i in range(len(datasets)):
    if i == 1:
        continue
    print("--------")
    print("DATASET: ", datasets[i])
    if datasets[i] == 'iris':
        data = load_iris()
        y = data.target
        y[y > 1] = 0
        y[y < 0] = 0
    elif datasets[i] == 'diabetes':
        data = load_diabetes()
        y = data.target
        y[y <= 100] = 0
        y[y > 100] = 1
    else:
        continue

    Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y, stratify=y)

    clf = RuleListClassifier(max_iter=50000, n_chains=3, class1label=data_class1_labels[i], verbose=True)

    clf.fit(Xtrain, ytrain, feature_labels=data_feature_labels[i])

    print("accuracy:", clf.score(Xtest, ytest))
    # print("rules:\n", clf)
    # print("Random Forest accuracy:", sklearn.ensemble.RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest))

    # clf.predict(Xtest)
    print(clf.predict_proba(Xtest))
    print(clf)