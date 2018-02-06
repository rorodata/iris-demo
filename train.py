"""Train a decision-tree model on the iris dataset.
"""
from __future__ import print_function, division

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import joblib

def find_score(model, X, y):
    """Computes the score as number of correct classifications
    by the sample size.
    """
    y2 = model.predict(X)
    # number of correct classfications
    correct = np.sum(y==y2)
    return correct/len(y)

def train(X, y):
    print("training the decision tree model")
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    print("accuracy is", find_score(model, X, y))
    return model

def save_model(model, filename):
    print("saving the model")
    joblib.dump(model, filename)

def main():
    print("loading iris dataset")
    iris = load_iris()
    X = iris['data']
    y = iris['target']

    model = train(X, y)
    save_model(model, "iris_model.pkl")

if __name__ == "__main__":
    main()
