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
from sklearn.decomposition import PCA

# Getting all data and initializing seed
f = open('PickleFiles/input.pckl', 'rb')
x_data = pickle.load(f)
f.close()

f = open('PickleFiles/one_hot_labels.pckl', 'rb')
y_data = pickle.load(f)
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

x_data = deleteFeatures(False, False, False, False, False, False, x_data)

smt = SMOTE()
x_data, y_data = smt.fit_sample(x_data, y_data)

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

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def model_with_cross_val(X_train, X_test, y_train, y_test):
    def ann(X_train):
        initializer = keras.initializers.he_uniform(seed=None) #0.8330
        model = Sequential()
        model.add(Dense(50, input_dim=len(X_train[0]), init=initializer, activation='relu'))
        model.add(Dense(25, init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto', restore_best_weights=True) # pateince is number of epochs
    callbacks_list = [earlystop]
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    model_infos = []
    metrics = []
    model = None
    for i,(train, test) in enumerate(kfold):
        model = None
        model = ann(X_train)
        print("Fit fold", i+1," ==========================================================================")
        model_info=model.fit(X_train[train], y_train[train], epochs=50, batch_size=32, validation_data=(X_train[test], y_train[test]),
                                verbose=0)
        print("Performance plot of fold {}:".format(i+1))
        #Final evaluation of the model
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

def model_without_cross_val(X_train, X_test, y_train, y_test):
    initializer = keras.initializers.he_uniform(seed=None) #0.8330
    model = Sequential()
    model.add(Dense(50, input_dim=len(X_train[0]), init=initializer, activation='relu'))
    model.add(Dense(25, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_info=model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
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

model_without_cross_val(X_train, X_test, y_train, y_test)