# Format: 
# 1) Imports
# 2) Load Training Set and CSV Files
# 3) Train Model
# 4) Test Model
# 5) Create Kaggle Submission

############################## IMPORTS ##############################

from __future__ import division
import sklearn
import pandas as pd
import numpy as np
import collections
import os.path
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingRegressor
import math
import csv
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import urllib
from sklearn.svm import LinearSVC

############################## LOAD TRAINING SET ##############################

if os.path.exists("Data/PrecomputedMatrices/xTrain.npy") and os.path.exists("Data/PrecomputedMatrices/yTrain.npy"):
	xTrain = np.load("Data/PrecomputedMatrices/xTrain.npy")
	yTrain = np.load("Data/PrecomputedMatrices/yTrain.npy")
	print ("Shape of xTrain:", xTrain.shape)
	print ("Shape of yTrain:", yTrain.shape)
else:
	print ('We need a training set! Run dataPreprocessing.py')
	sys.exit()

############################## LOAD CSV FILES ##############################

sample_sub_pd = pd.read_csv('Data/KaggleData/SampleSubmissionStage1.csv')

############################## TRAIN MODEL ##############################

model1 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.1)
model = CalibratedClassifierCV(model1) 

categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
           'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','Location']
accuracy=[]
numTrials = 1

for i in range(numTrials):
    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
    results = model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    localAccuracy = np.mean(preds == Y_test)
    accuracy.append(localAccuracy)
    print ("Finished run #" + str(i) + ". Accuracy = " + str(localAccuracy))
print ("The average accuracy is", sum(accuracy)/len(accuracy))

############################## TEST MODEL ##############################

def predictGame(team_1_vector, team_2_vector, home):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)
    # Depending on the model you use, you will either need to return model.predict_proba or model.predict
    # predict_proba = Linear Reg, Linear SVC
    # predict = Gradient Boosted

    return model.predict_proba([diff])[0][1]
    #return model.predict([diff])[0]

############################## CREATE KAGGLE SUBMISSION ##############################

def loadTeamVectors(years):
	listDictionaries = []
	for year in years:
		curVectors = np.load("Data/PrecomputedMatrices/TeamVectors/" + str(year) + "TeamVectors.npy").item()
		listDictionaries.append(curVectors)
	return listDictionaries

def createPrediction():
	if os.path.exists("result.csv"):
		os.remove("result.csv")
	# The years that we want to predict for
	years = range(2014,2018)
	listDictionaries = loadTeamVectors(years)
	print ("Loaded the team vectors")
	results = [[0 for x in range(2)] for x in range(len(sample_sub_pd.index))]
	for index, row in sample_sub_pd.iterrows():
		matchupId = row['ID']
		year = int(matchupId[0:4]) 
		teamVectors = listDictionaries[year - years[0]]
		team1Id = int(matchupId[5:9])
		team2Id = int(matchupId[10:14])
		team1Vector = teamVectors[team1Id] 
		team2Vector = teamVectors[team2Id]
		pred = predictGame(team1Vector, team2Vector, 0)
		results[index][0] = matchupId
		results[index][1] = pred
	results = pd.np.array(results)
	firstRow = [[0 for x in range(2)] for x in range(1)]
	firstRow[0][0] = 'ID'
	firstRow[0][1] = 'Pred'
	with open("result.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(firstRow)
		writer.writerows(results)

createPrediction()