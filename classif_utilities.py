# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 03/10/2016
# ----------------------------------------

import pyriemann as pr
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from pyriemann.classification import TSclassifier
from pyriemann.classification import MDM


def split_dataset(X, y, indices=False):
    """
    Split a dataset in train/validation sets regarding to the class balance
    Output training and validation sets are representatives of the original dataset in term of class balance
    Output validation set length is about 1/3 of the input dataset length
    :param X: input variables
    :param y: target variables
    :return: training set (part of X and y) and validation set (complementary part of
            training set) ==> X_train, y_train, X_valid, y_valid
            returns also the list of indices of the validation set when indices parameter is True
    """
    skf = StratifiedKFold(y, n_folds=3, random_state=42, shuffle=True)
    for i, (train_test_indices, valid_indices) in enumerate(skf):
        if i == 0:
            break
    X_train_test = X[train_test_indices]
    y_train_test = y[train_test_indices]
    X_valid = X[valid_indices]
    y_valid = y[valid_indices]
    if not indices:
        return X_train_test, y_train_test, X_valid, y_valid
    else:
        return X_train_test, y_train_test, X_valid, y_valid, valid_indices


def cross_valid_clf(clf, X_train, y_train, n_folds=5):
    """
    Cross validate with a classifier with StratifiedKFolds to keep a representative balance of classes
    :param clf: classifier (inherited from BaseEstimator)
    :param X_train: training input
    :param y_train: training output
    :param n_folds: number of folds (default=5)
    :return: accuracy scores for each validation
    """
    cv = StratifiedKFold(y_train, n_folds=n_folds)
    accuracy = np.array([], dtype=float)
    for i, (train, test) in enumerate(cv):
        y_pred = clf.fit(X_train[train], y_train[train]).predict(X_train[test])
        acc = accuracy_score(y_train[test], y_pred)
        accuracy = np.append(accuracy, [acc])
    return accuracy


def validate_clf(clf, X_train, y_train, X_valid, y_valid):
    """
    Generate validation results for a classifier, a training set and a validation set
    :param clf: classifier (inherited from BaseEstimator and implementing a predict_proba method)
                input classifier should be fitted
    :param X_train: inputs of training set
    :param y_train: targets of training set
    :param X_valid: inputs of validation set
    :param y_valid: targets of validation set
    :return: accuracy, recall and roc_auc scores for validation set
    """
    clf.fit(X_train, y_train)
    y_valid_pred = clf.predict(X_valid)
    y_valid_proba = clf.predict_proba(X_valid)[:, 1]
    return accuracy_score(y_valid, y_valid_pred), recall_score(y_valid, y_valid_pred), \
           precision_score(y_valid, y_valid_pred), roc_auc_score(y_valid, y_valid_proba)


def roc_curves_4_clfs(clfs, X_valid, y_valid, labels):
    """
    Plot ROC curves for a list of fitted classifier and a validation set
    :param clfs: list of fitted classifiers
    :param X_valid: inputs of validation set
    :param y_valid: targets of validation set
    :param labels: list of labels for classifiers
    :return: None
    """
    colors = ['#40bf80', '#668cff', '#ffa64d', '#ff33bb', '#330033', '#4dffc3', '#805500', '#999900']
    plt.figure(figsize=(10, 7))
    for i, clf in enumerate(clfs):
        y_proba = clf.predict_proba(X_valid)[:, 1]
        fpr, tpr, _ = roc_curve(y_valid, y_proba)
        ras = roc_auc_score(y_valid, y_proba)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=labels[i] + " (AUC={0:.2f})".format(ras), color=colors[i % len(colors)])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')


if __name__ == "__main__":
    # Load Matrix and target
    X = np.load("data/icu_matrix2.pyriemann.npy")
    y = np.load("data/icu_target2.pyriemann.npy")
    X.shape, np.size(y)
    skf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)