import pickle
import numpy
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np

# Getting all data
f = open('PickleFiles/institution_1_input.pckl', 'rb')
inst_1_x = pickle.load(f)
f.close()

f = open('PickleFiles/institution_2_input.pckl', 'rb')
inst_2_x = pickle.load(f)
f.close()

f = open('PickleFiles/institution_1_labels.pckl', 'rb')
inst_1_y = pickle.load(f)
f.close()

f = open('PickleFiles/institution_2_labels.pckl', 'rb')
inst_2_y = pickle.load(f)
f.close()

f = open('PickleFiles/test_input.pckl', 'rb')
test_input = pickle.load(f)
f.close()

f = open('PickleFiles/test_labels.pckl', 'rb')
test_labels = pickle.load(f)
f.close()

# smt = SMOTE()
# x_data, y_data = smt.fit_sample()

count_cancer = 0
count_none = 0



# print(inst_1_y)
# print("#####")
# print(inst_2_y)
# print("#####")
# print(test_labels)
# print("#####")
# print(len(inst_1_x))
# print("#####")
# print(len(inst_2_x))
# print("#####")
# print(len(test_input))

for i in inst_2_y:
    print(i)


# print("Percent of Data Samples with Cancer: " + str(count_cancer/(count_cancer + count_none)))
# print("Percent of Data Samples with None: " + str(count_none/(count_cancer + count_none)))
# print("Number of Data Samples with Cancer: " + str(count_cancer))
# print("Number of Data Samples with None: " + str(count_none))

# scaler = StandardScaler()
# x = scaler.fit_transform(x_data)