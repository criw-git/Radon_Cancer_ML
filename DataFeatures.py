import pickle
import numpy
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Getting all data
f = open('PickleFiles/input.pckl', 'rb')
x_data = pickle.load(f)
f.close()

f = open('PickleFiles/labels.pckl', 'rb')
y_data = pickle.load(f)
f.close()

smt = SMOTE()
x_data, y_data = smt.fit_sample(x_data, y_data)

count_cancer = 0
count_none = 0

for i in y_data:
    if (i == 0):
        count_cancer += 1
    if (i == 1):
        count_none += 1

print("Percent of Data Samples with Cancer: " + str(count_cancer/(count_cancer + count_none)))
print("Percent of Data Samples with None: " + str(count_none/(count_cancer + count_none)))
print("Number of Data Samples with Cancer: " + str(count_cancer))
print("Number of Data Samples with None: " + str(count_none))

scaler = StandardScaler()
x = scaler.fit_transform(x_data)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])

# fields = ['SEX', 'AGE', 'RACE', 'SMOKING_RISK', 'DX_CODE', 'RADON_RISK']
# df = pd.read_csv('smoking_radon_deid2.csv', usecols = fields)

# finalDf = pd.concat([principalDf, df[['DX_CODE']]], axis = 1)
# print("#################################")
# print(finalDf)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = ['cancer', 'Other', 'COPD']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['DX_CODE'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()

# plt.show()