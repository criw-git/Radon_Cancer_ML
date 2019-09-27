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

fields = ['PAT_MRN_ID', 'SEX', 'AGE', 'RACE', 'ZIP', 'SMOKING_RISK', 'DX_CODE', 'RADON_RISK']
# Total of 6302 data points
df1 = pd.read_csv('Data_Request_Final.csv', usecols = fields)

fields = ['PAT_MRN_ID', 'decile_pscore']
# Total of 4437 data points
df2 = pd.read_csv('pscore_umit_sep2019.csv', usecols = fields)

df = pd.merge(df1, df2, left_on='PAT_MRN_ID', right_on='PAT_MRN_ID')
df = df.dropna()

df['DX_CODE'] = np.where(df['DX_CODE'].str.contains('^C34'), 'cancer', df['DX_CODE'])
df['DX_CODE'] = np.where(df['DX_CODE'].str.contains('^J44'), 'cancer', df['DX_CODE'])
df['DX_CODE'] = np.where(df['DX_CODE'].str.contains('162'), 'cancer', df['DX_CODE'])
df['DX_CODE'] = np.where(df['DX_CODE'].str.contains('1'), 'Other', df['DX_CODE'])
df['DX_CODE'] = np.where(df['DX_CODE'].str.contains('2'), 'Other', df['DX_CODE'])
df['DX_CODE'] = np.where(df['DX_CODE'].str.contains('9'), 'Other', df['DX_CODE'])


# Converting all the features to lists
sex_list = (df.SEX).values.tolist() # there is nan
age_list = (df.AGE).values.tolist() 
race_list = (df.RACE).values.tolist()
smoking_risk_list = (df.SMOKING_RISK).values.tolist()
radon_risk_list = (df.RADON_RISK).values.tolist()
zip_list = (df.ZIP).values.tolist()
pscore_list = (df.decile_pscore).values.tolist() # has nan

# Converting labels to lists
dx_code_list = (df.DX_CODE).values.tolist()

def list_to_vec(sex_list, race_list, dx_code_list, zip_list):
    a = []
    for i in range(len(sex_list)):
        if (sex_list[i] == 'Male'):
            a.append(0)
        else:
            a.append(1)
    b = []
    for i in range(len(race_list)):
        if (race_list[i] == 'Other'):
            b.append(0)
        elif (race_list[i] == 'Asian'):
            b.append(1)
        elif (race_list[i] == 'Patient Refused'):
            b.append(2)
        elif (race_list[i] == 'American Indian or Alaska Native'):
            b.append(3)
        elif (race_list[i] == 'White or Caucasian'):
            b.append(4)
        elif (race_list[i] == 'Unknown'):
            b.append(5)
        elif (race_list[i] == 'Black or African American'):
            b.append(6)
    c = []
    for i in range(len(dx_code_list)):
        if (dx_code_list[i] == 'cancer'):
            c.append(0)
        elif (dx_code_list[i] == 'Other'):
            c.append(1)
        elif (dx_code_list[i] == 'COPD'):
            c.append(0)
    d = []
    le = preprocessing.LabelEncoder()
    le.fit(zip_list)
    d = le.transform(zip_list)
    return a, b, c, d

sex_list, race_list, dx_code_list, zip_list = list_to_vec(sex_list, race_list, dx_code_list, zip_list)

input_data = []
def create_input_data(sex_list, age_list, race_list, smoking_risk_list, radon_risk_list, zip_list, input_data, pscore_list):
    for i in range(len(sex_list)):
        patient = []
        patient.append(sex_list[i])
        patient.append(age_list[i])
        patient.append(race_list[i])
        patient.append(smoking_risk_list[i])
        patient.append(radon_risk_list[i])
        patient.append(zip_list[i])
        patient.append(pscore_list[i])
        input_data.append(patient)
    return input_data

x_data = create_input_data(sex_list, age_list, race_list, smoking_risk_list, radon_risk_list, zip_list, input_data, pscore_list)

labels = np.array(dx_code_list)
l = np_utils.to_categorical(labels)
y_data = l

#Splitting for institutions and then processing
data = []
def create_all_data(sex_list, age_list, race_list, smoking_risk_list, radon_risk_list, zip_list, dx_code_list, pscore_list, data):
    for i in range(len(sex_list)):
        patient = []
        patient.append(sex_list[i])
        patient.append(age_list[i])
        patient.append(race_list[i])
        patient.append(smoking_risk_list[i])
        patient.append(radon_risk_list[i])
        patient.append(zip_list[i])
        patient.append(pscore_list[i])
        patient.append(dx_code_list[i])
        data.append(patient)
    return data
data = create_all_data(sex_list, age_list, race_list, smoking_risk_list, radon_risk_list, zip_list, dx_code_list, pscore_list, data)

rn.shuffle(data)
marker = round(len(data) * 0.4)
num_institutions = [1, 2]

def create_institusional_data(data, num_institutions, marker):
    all_institutional_data = []
    for i in num_institutions:
        theData = data[((i - 1) * ( marker )) : (i * marker)]
        all_institutional_data.append(theData)
    testMarker = marker * num_institutions[len(num_institutions) - 1] + 1
    test = data[testMarker:]

    def seperate_data(institution_data):
        new_input_data = []
        new_output_data = []
        for data in institution_data:
            patient_output = []
            patient_input = []
            patient_output.append(data[6])
            patient_input.append(data[0])
            patient_input.append(data[1])
            patient_input.append(data[2])
            patient_input.append(data[3])
            patient_input.append(data[4])
            patient_input.append(data[5])
            new_input_data.append(patient_input)
            new_output_data.append(patient_output)
        return new_input_data, new_output_data
    input_institutional_data = []
    labels_institutional_data = []
    for i in all_institutional_data:
        institution_input, institution_labels = seperate_data(i)
        input_institutional_data.append(institution_input)
        labels_institutional_data.append(institution_labels)
    test_input, test_labels = seperate_data(test)

    input_institutional_data2 = []
    for i in input_institutional_data:
        input_institutional_data2.append(np.array(i))
    test_input = np.array(test_input)

    labels_institutional_data2 = []
    for i in labels_institutional_data:
        labels = np.array(i)
        l = np_utils.to_categorical(labels)
        labels_institutional_data2.append(l)

    test_labels = np.array(test_labels)
    l = np_utils.to_categorical(test_labels)
    test_labels = l

    return input_institutional_data2, labels_institutional_data2, test_input, test_labels

input_institutional_data2, labels_institutional_data2, test_input, test_labels = create_institusional_data(data, num_institutions, marker)

#Subject to change according to number of institutions
institution_1_input = input_institutional_data2[0]
institution_2_input = input_institutional_data2[1]
institution_1_labels = labels_institutional_data2[0]
institution_2_labels = labels_institutional_data2[1]

# Saving x and y data
def save_all_variables(x_data, y_data, labels, institution_1_input, institution_2_input, institution_1_labels, institution_2_labels, test_input, test_labels):
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

    f = open('SingleWeightTransfer/PickleFiles/institution_1_input.pckl', 'wb')
    pickle.dump(institution_1_input, f)
    f.close()
    print("Saved institution 1 input data")

    f = open('SingleWeightTransfer/PickleFiles/institution_2_input.pckl', 'wb')
    pickle.dump(institution_2_input, f)
    f.close()
    print("Saved institution 2 input data")

    f = open('SingleWeightTransfer/PickleFiles/institution_1_labels.pckl', 'wb')
    pickle.dump(institution_1_labels, f)
    f.close()
    print("Saved institution 1 labels")

    f = open('SingleWeightTransfer/PickleFiles/institution_2_labels.pckl', 'wb')
    pickle.dump(institution_2_labels, f)
    f.close()
    print("Saved institution 2 labels")

    f = open('SingleWeightTransfer/PickleFiles/test_input.pckl', 'wb')
    pickle.dump(test_input, f)
    f.close()
    print("Saved test input data")

    f = open('SingleWeightTransfer/PickleFiles/test_labels.pckl', 'wb')
    pickle.dump(test_labels, f)
    f.close()
    print("Saved test labels")

    f = open('CyclicalWeightTransfer/PickleFiles/institution_1_input.pckl', 'wb')
    pickle.dump(institution_1_input, f)
    f.close()
    print("Saved institution 1 input data")

    f = open('CyclicalWeightTransfer/PickleFiles/institution_2_input.pckl', 'wb')
    pickle.dump(institution_2_input, f)
    f.close()
    print("Saved institution 2 input data")

    f = open('CyclicalWeightTransfer/PickleFiles/institution_1_labels.pckl', 'wb')
    pickle.dump(institution_1_labels, f)
    f.close()
    print("Saved institution 1 labels")

    f = open('CyclicalWeightTransfer/PickleFiles/institution_2_labels.pckl', 'wb')
    pickle.dump(institution_2_labels, f)
    f.close()
    print("Saved institution 2 labels")

    f = open('CyclicalWeightTransfer/PickleFiles/test_input.pckl', 'wb')
    pickle.dump(test_input, f)
    f.close()
    print("Saved test input data")

    f = open('CyclicalWeightTransfer/PickleFiles/test_labels.pckl', 'wb')
    pickle.dump(test_labels, f)
    f.close()
    print("Saved test labels")

save_all_variables(x_data, y_data, labels, institution_1_input, institution_2_input, institution_1_labels, institution_2_labels, test_input, test_labels)