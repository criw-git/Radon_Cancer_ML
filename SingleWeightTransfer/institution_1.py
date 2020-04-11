import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import random as rn
import os
import random
import keras
import statistics as st
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from joblib import load, dump

# Getting all data and initializing seed
f = open('PickleFiles/institution_1_input.pckl', 'rb')
X_train = pickle.load(f)
f.close()

f = open('PickleFiles/institution_1_labels.pckl', 'rb')
y_train = pickle.load(f)
f.close()

f = open('PickleFiles/test_input.pckl', 'rb')
X_test = pickle.load(f)
f.close()

f = open('PickleFiles/test_labels.pckl', 'rb')
y_test = pickle.load(f)
f.close()

seed = 42

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

X_train = deleteFeatures(False, False, False, False, False, False, X_train)
X_test = deleteFeatures(False, False, False, False, False, False, X_test)

smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)

# Function to get metrics
def evaluate_model(metrics, model, y_test, X_test):
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    metrics.append(metric)
    return metrics, y_pred

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def perform_single_institution(X_train, X_test, y_train, y_test):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True) # pateince is number of epochs
    callbacks_list = [earlystop]
    initializer = keras.initializers.he_uniform(seed=None) #0.8330
    model = Sequential()
    model.add(Dense(50, input_dim=len(X_train[0]), init=initializer, activation='relu'))
    model.add(Dense(25, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_info=model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=callbacks_list)
    #Final evaluation of the model
    metrics = []
    metrics, y_pred = evaluate_model(metrics, model, y_test, X_test)

    def findAverage(all_metrics):
        f1 = []
        precision = []
        recall = []
        accuracy = []
        avg_metrics = []
        for metrics in all_metrics:
            f1.append(metrics[0][1])
            precision.append(metrics[1][1])
            recall.append(metrics[2][1])
            accuracy.append(metrics[3][1])
        avg_metrics.append(['f1score',np.mean(f1)])
        avg_metrics.append(['precision',np.mean(precision)])
        avg_metrics.append(['recall',np.mean(recall)])
        avg_metrics.append(['accuracy',np.mean(accuracy)])
        
        return avg_metrics

    avg_metrics = findAverage(metrics)
    print("Average Scores")
    print(avg_metrics)
    return model

model = perform_single_institution(X_train, X_test, y_train, y_test)
model.save('Models/ann_inst1_model')

#################################################################################################################
# Extra Trees Classifier - create model and fit
def ExtraTrees(X_train, X_test, y_train, y_test):
    model = ExtraTreesClassifier(random_state=0, n_estimators=150, bootstrap=True, oob_score=True, warm_start=True)
    model.fit(X_train, y_train)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    metrics = []
    metrics.append(['f1score',f1_score(y_pred, y_test)])
    #metrics.append(['precision',precision_score(y_pred, y_test)])
    #metrics.append(['recall',recall_score(y_pred, y_test)])
    metrics.append(['accuracy',accuracy_score(y_pred, y_test)])
    print(metrics) 
    return model

model = ExtraTrees(X_train, X_test, y_train, y_test)
dump(model, 'Models/exttree_inst1_model.joblib')
