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
part-2
'''
path_to_testing_dat = sys.argv[1]
path_to_classifier = sys.argv[2]
path_to_results = sys.argv[3]
# testing_data = []
with open(path_to_testing_dat, 'rb') as f:

	testing_data = pickle.load(f)

with open(path_to_classifier, 'rb') as f:

	classifier = pickle.load(f)

with open('vector.dat','rb') as f:
	
	v = pickle.load(f)

list_test = []
words = []

for i in range(len(testing_data)):
	for j in range(len(testing_data[i])):

		word = {}

		words.append(testing_data[i][j])

		mark = testing_data[i][j][0]

		word["ABC"] = mark

		word["DEFG"] = testing_data[i][j][1]

		if(j == 0):
			word["A"] = testing_data[i][j + 1][0]
			word["ATag"] = testing_data[i][j + 1][1]

		elif(j == len(testing_data[i]) - 1):
			word["B"] = testing_data[i][j - 1][0]
			word["BTag"] = testing_data[i][j - 1][1]

		elif(j < len(testing_data[i]) - 1):
			word["A"] = testing_data[i][j + 1][0]
			word["ATag"] = testing_data[i][j + 1][1]

			word["B"] = testing_data[i][j - 1][0]
			word["BTag"] = testing_data[i][j - 1][1]	

		list_test.append(word)


code_test = v.transform(list_test)

result= classifier.predict(code_test)
# print (result)

with open(path_to_results, 'wb') as f:
    pickle.dump(result, f)

'''
x= classifier.predict(code_test)
#print (x)

# print (testing_data)
list_1 = []
for i in testing_data:
	# print (i)
	for j in i:
		# print (j)
		list_1.append(j)
		# for k in range(0,len(x)):

		# 	# print (x[k])
		# 	j.append(x[k])
		# print (j)
		# print ('\n')


#list_1 是原数据集，x是预测结果
for i in range(0,len(list_1)):
	list_1[i].append(x[i])
#打印
# for i in list_1:
# 	print ('___________________')
# 	print (i)
# print (list_1[1000])
# print (list_1[1000][2])
# print (list_1[1000][3])
#对比
tp=0
fp=0
fn=0
for i in list_1:
	# print (i)
	if i[2]== 'TITLE' and i[3]== 1:
		tp=tp+1
	if i[2]== 'O' and i[3]==1:
		fp=fp+1
	if i[2]=='TITLE' and i[3]==0:
		fn=fn+1
	
	# elif i[2]=='TITLE' and i[3]!='1':
		# print('3')
		
# print ('tp = ',tp)
# print ('fp = ',fp)
# print ('fn = ',fn)

r=tp/(tp + fn)
p = tp / (tp + fp)
F1 = 2*p*r/(p+r)

print ('recall : ',r,'   precision : ',p)
print ('F1-measure : ',F1)
'''



