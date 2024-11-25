

from sklearn.svm import OneClassSVM
import numpy as np
np.set_printoptions(suppress = True)
import pandas as pd
from EER import evaluateEER
from fancyimpute import IterativeImputer
import sys
from sklearn.impute import SimpleImputer

class SVMDetector:
#just the training() function changes, rest all remains same.

    def __init__(self, users):
        self.u_scores = []
        self.i_scores = []
        self.mean_vector = []
        self.users = users
        
    
    def training(self):
        self.clf = OneClassSVM(kernel='rbf',gamma=0.005)
        self.clf.fit(self.train)
 
    def testing(self):
        self.u_scores = -self.clf.decision_function(self.test_genuine)
        self.i_scores = -self.clf.decision_function(self.test_imposter)
        self.u_scores = list(self.u_scores)
        self.i_scores = list(self.i_scores)
 
    def evaluate(self):
        eers = []
 
        for user in users:        
            genuine_user_data = datai.loc[datai.User == user, \
                                         "?2_Latency":"TP_Latency"]
            imposter_data = datai.loc[datai.User != user, :]
            
            self.train = genuine_user_data[:60]
            self.test_genuine = genuine_user_data[60:]
            self.test_imposter = imposter_data.groupby("User"). \
                                 head(5).loc[:, "?2_Latency":"TP_Latency"]
 
            self.training()
            self.testing()
            eers.append(evaluateEER(self.u_scores, \
                                     self.i_scores))
        print(eers)                             
        return np.mean(eers)



path = "keystroke.csv"
#path = "keynew.csv" 
datai = pd.read_csv(path, header=0, na_values='nan')

m = round(datai.mean(axis=1),3)

for i, col in enumerate(datai):
	# using i allows for duplicate columns
	# inplace *may* not always work here, so IMO the next line is preferred
	#datai.iloc[:, i].fillna(m, inplace=True)
	datai.iloc[:, i] = datai.iloc[:, i].fillna(m)


datai.to_csv(r'keymean.csv', index=False)             

print(datai.head(5))

datai1 = datai.iloc[:366,:] 
datai2 = datai.iloc[366:732,:]
datai3 = datai.iloc[732:1098,:]
datai4 = datai.iloc[1098:1464,:]
datai5 = datai.iloc[1464:,:]

users = datai1["User"].unique()
print ("average EER for fold 1 SVM detector:")
print(SVMDetector(users).evaluate())

users = datai2["User"].unique()
print ("average EER for fold 2 SVM detector:")
print(SVMDetector(users).evaluate())

users = datai3["User"].unique()
print ("average EER for fold 3 SVM detector:")
print(SVMDetector(users).evaluate())

users = datai4["User"].unique()
print ("average EER for fold 4 SVM detector:")
print(SVMDetector(users).evaluate())

users = datai5["User"].unique()
print ("average EER for fold 5 SVM detector:")
print(SVMDetector(users).evaluate())


