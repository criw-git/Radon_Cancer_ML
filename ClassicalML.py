import numpy as np
import tensorflow as tf
import random as rn
import os
import random
import statistics as st
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from xgboost import XGBClassifier

# Getting all data
f = open('PickleFiles/input.pckl', 'rb')
x_data = pickle.load(f)
f.close()

f = open('PickleFiles/labels.pckl', 'rb')
y_data = pickle.load(f)
f.close()

def deleteFeatures(sex, age, race, smoking, radon, zipc, x_data):
    x_data = list(x_data)
    b = []
    for data in x_data:
        a = list(data)
        if (sex):
            del a[0]
        if (age):
            del a[1]
        if (race):
            del a[2]
        if (smoking):
            del a[3]
        if (radon):
            del a[4]
        if (zipc):
            del a[5]
        a = np.array(a)
        b.append(a)
    b = np.array(b)
    return b

x_data = deleteFeatures(False, False, False, False, False, False, x_data)

smt = SMOTE()
x_data, y_data = smt.fit_sample(x_data, y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state = 42)
###############################################################################################

# Logistic Regression - create model and fit
def logisticRegression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=None, max_iter=900, solver='newton-cg', penalty='l2')
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making logistic regression model")
metrics, model = logisticRegression(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# Naive Bayes - create model and fit
def naiveBayes(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making Naive Bayes model")
metrics, model = naiveBayes(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# Support Vector Machine - create model and fit
def SVM(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    model = SVC(kernel = 'rbf', C = 1)
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test_scaled)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making Support Vector Machine model")
metrics, model = SVM(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# K-nearest neighbors - create model and fit
def KNN(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='brute')
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making KNN model")
metrics, model = KNN(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# Decision Tree - create model and fit
def DecisionTree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=0, presort=True)
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making Decision Tree model")
metrics, model = DecisionTree(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# Random Forest - create model and fit
def RandomForest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=50, random_state=0, bootstrap=True, oob_score=True, warm_start=True)
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making Random Forest model")
metrics, model = RandomForest(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# Extra Trees Classifier - create model and fit
def ExtraTrees(X_train, X_test, y_train, y_test):
    model = ExtraTreesClassifier(random_state=0, n_estimators=100)
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making Extra Trees model")
metrics, model = ExtraTrees(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# Gradient Boosting - create model and fit
def GradientBoost(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(random_state=0, warm_start=True, loss="deviance", n_estimators=400)
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making Gradient Boosting model")
metrics, model = GradientBoost(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# MLP (Multi-Peceptron Layer) Classifier - create model and fit (technically deep learning)
def MLP(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    model = MLPClassifier(random_state=0, max_iter=300, early_stopping=False)
    model.fit(X_train_scaled, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test_scaled)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making Multi-Perceptron Layer model")
metrics, model = MLP(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################

# XGB - create model and fit
def XGB(X_train, X_test, y_train, y_test):
    model = XGBClassifier(random_state=0)
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    return metrics, model

print("Making XGB model")
metrics, model = XGB(X_train, X_test, y_train, y_test)
print(metrics)
print('################')
###############################################################################################