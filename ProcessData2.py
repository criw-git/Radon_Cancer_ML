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
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

fields = ['SEX', 'AGE', 'RACE', 'SMOKING_RISK', 'DX_CODE', 'RADON_RISK']

# Total of 6302 data points
df = pd.read_csv('SPARC13743_MUSC_COMBINED_data.csv', usecols = fields)
#df = pd.read_csv('smoking_radon_deid2.csv', usecols = fields)

# Converting all the features to lists
sex_list = (df.SEX).values.tolist()
age_list = (df.AGE).values.tolist()
race_list = (df.RACE).values.tolist()
smoking_risk_list = (df.SMOKING_RISK).values.tolist()
radon_risk_list = (df.RADON_RISK).values.tolist()
#zip_list = (df.ZIP).values.tolist()

# Converting labels to lists
dx_code_list = (df.DX_CODE).values.tolist()

# function to get unique values 
def unique(list1):  
    list_set = set(list1) 
    unique_list = (list(list_set)) 
    for x in unique_list: 
        print(x)

print(unique(sex_list))

def list_to_vec(sex_list, race_list, dx_code_list):
    a = []
    for i in range(len(sex_list)):
        if (sex_list[i] == 'Male'):
            a.append(0)
        elif (sex_list[i] == 'Female'):
            a.append(1)
        else:
            a.append(2)
    b = []
    for i in range(len(race_list)):
        if (race_list[i].lower() == 'other'):
            b.append(0)
        elif (race_list[i].lower() == 'asian'):
            b.append(1)
        elif (race_list[i].lower() == 'patient refused'):
            b.append(2)
        elif (race_list[i].lower() == 'american indian or alaska native'):
            b.append(3)
        elif (race_list[i].lower() == 'white or caucasian'):
            b.append(4)
        elif (race_list[i].lower() == 'unknown'):
            b.append(5)
        elif (race_list[i].lower() == 'black or african american'):
            b.append(6)
        elif (race_list[i].lower() == 'native hawaiian or other pacific islander'):
            b.append(7)
    c = []
    for i in range(len(dx_code_list)):
        if (dx_code_list[i] == 'cancer'):
            c.append(0)
        elif (dx_code_list[i] == 'Other'):
            c.append(1)
        elif (dx_code_list[i] == 'COPD'):
            c.append(0)
    # d = []
    # le = preprocessing.LabelEncoder()
    # le.fit(zip_list)
    # d = le.transform(zip_list)
    return a, b, c

sex_list, race_list, dx_code_list= list_to_vec(sex_list, race_list, dx_code_list)

input_data = []
def create_input_data(sex_list, age_list, race_list, smoking_risk_list, radon_risk_list, input_data):
    for i in range(len(sex_list)):
        patient = []
        patient.append(sex_list[i])
        patient.append(age_list[i])
        patient.append(race_list[i])
        patient.append(smoking_risk_list[i])
        patient.append(radon_risk_list[i])
        #patient.append(zip_list[i])
        input_data.append(patient)
    return input_data

x_data = create_input_data(sex_list, age_list, race_list, smoking_risk_list, radon_risk_list, input_data)
x_data = np.array(x_data)

# One-hot code the labels
labels = np.array(dx_code_list)
l = np_utils.to_categorical(labels)
y_data = l


# Saving x and y data
def save_all_variables(x_data, y_data, labels):
    f = open('PickleFiles/input.pckl', 'wb')
    pickle.dump(x_data, f)
    f.close()
    print("Saved input data")

    f = open('PickleFiles/one_hot_labels.pckl', 'wb')
    pickle.dump(y_data, f)
    f.close()
    print("Saved one hot labels data")

    f = open('PickleFiles/labels.pckl', 'wb')
    pickle.dump(labels, f)
    f.close()
    print("Saved labels data")

save_all_variables(x_data, y_data, labels)