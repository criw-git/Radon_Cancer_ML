import warnings
warnings.filterwarnings("ignore")

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
from vecstack import stacking
from vecstack import StackingTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import VotingClassifier
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
##############################################################################################
# Stacking three models
models = [ExtraTreesClassifier(random_state=0, n_estimators=150, bootstrap=True, oob_score=True, warm_start=True),
             RandomForestClassifier(n_estimators=75, random_state=0, bootstrap=True, oob_score=True, warm_start=True), 
             DecisionTreeClassifier(random_state=0, presort=True), 
             XGBClassifier(random_state=0)]

S_train, S_test = stacking(models, X_train, y_train, X_test, regression=False, mode='oof_pred_bag', 
                    needs_proba=False, save_dir=None, metric=accuracy_score, n_folds=4, stratified=True,
                    shuffle=True, random_state=0, verbose=1)

model = GradientBoostingClassifier(random_state=0, warm_start=True, loss="deviance", n_estimators=400)
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)
metrics = []
metrics.append(['f1score',f1_score(y_pred, y_test)])
metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
print(metrics)
print("#################################################################################")

#Increased performance a bit
##############################################################################################

# Hard Voting Classifier with three models
clf1 = ExtraTreesClassifier(random_state=0, n_estimators=150, bootstrap=True, oob_score=True, warm_start=True)
clf2 = DecisionTreeClassifier(random_state=0, presort=True)
clf3 = RandomForestClassifier(n_estimators=50, random_state=0, bootstrap=True, oob_score=True, warm_start=True)
clf4 = LogisticRegression(random_state=None, max_iter=900, solver='newton-cg', penalty='l2')

eclf = VotingClassifier(estimators=[('et', clf1), ('dt', clf2), ('rf', clf3), ('lr', clf4)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Extra Trees Classifier', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'Ensemble']):
    scores = cross_val_score(clf, x_data, y_data, cv=5, scoring='accuracy')
    print("Accuracy: " + str(scores.mean()) + " +/- " + str(scores.std()) + " " + label)

print("#################################################################################")
#Acheived 91% accuracy with margin when models were extra trees, decision tree, random forest, logistic regression
##############################################################################################

# Soft Voting Classifier with three models
clf1 = ExtraTreesClassifier(random_state=0, n_estimators=150, bootstrap=True, oob_score=True, warm_start=True)
clf2 = DecisionTreeClassifier(random_state=0, presort=True)
clf3 = RandomForestClassifier(n_estimators=50, random_state=0, bootstrap=True, oob_score=True, warm_start=True)
clf4 = LogisticRegression(random_state=None, max_iter=900, solver='newton-cg', penalty='l2')
clf5 = GradientBoostingClassifier(random_state=0, warm_start=True, loss="deviance", n_estimators=400)

eclf = VotingClassifier(estimators=[('et', clf1), ('dt', clf2), ('rf', clf3), ('lr', clf4), ('gb', clf5)], voting='soft', weights=[2, 1, 2, 2, 1])

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Extra Trees Classifier', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'Gradient Boosting', 'Ensemble']):
    scores = cross_val_score(clf, x_data, y_data, cv=5, scoring='accuracy')
    print("Accuracy: " + str(scores.mean()) + " +/- " + str(scores.std()) + " " + label)

print("#################################################################################")
#Didn't really increase performance
##############################################################################################

