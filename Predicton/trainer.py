import sklearn
import numpy
import scipy
import pickle
import os
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

'''
part-1
'''
data_path = sys.argv[1]
classifier_path = sys.argv[2]

target=[]

words_dicts= []

outcomes = ['O','TITLE']

with open(data_path, 'rb') as f:         #"/Users/danielsapple/Desktop/9318/project/training_data.dat"	
	training_set = pickle.load(f)
	
	classifier = linear_model.LogisticRegression(class_weight="balanced",max_iter=200)

	D=[]

	E=[]

for i in range(len(training_set)):
	for j in range(len(training_set[i])):
		word = {}

		mark = training_set[i][j][0]

		word["ABC"] = mark

		word["DEFG"] = training_set[i][j][1]

		if(j == 0):
			word["A"] = training_set[i][j + 1][0]

			word["ATag"] = training_set[i][j + 1][1]

		elif(j == len(training_set[i]) - 1):
			word["B"] = training_set[i][j - 1][0]

			word["BTag"] = training_set[i][j - 1][1]

		elif(j < len(training_set[i]) - 1):
			word["A"] = training_set[i][j + 1][0]

			word["ATag"] = training_set[i][j + 1][1]

			word["B"] = training_set[i][j - 1][0]

			word["BTag"] = training_set[i][j - 1][1]

		if(training_set[i][j][2] == "TITLE"):
			E.append(1)

		else:
			E.append(0)

		D.append(word)

#print (word)

y=[]

for sentence in training_set:

	k=len(sentence)

	for i in range(0,len(sentence)):

		 word_dict = {}

		 word = sentence[i]

		 target.append(outcomes.index(word[2]))	 

		 word_dict[word[0]] = 1

		 words_dicts.append(word_dict)

v = DictVectorizer(sparse = False)
X = v.fit_transform(D)

classifier.fit(X, E)

with open(classifier_path, 'wb') as f:
	pickle.dump(classifier, f)

with open('vector.dat','wb') as f:
	pickle.dump(v,f)

