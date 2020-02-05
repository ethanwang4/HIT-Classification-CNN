#Imports
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from random import randint
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

#Loading and parsing MAUDE reports
with open("MAUDE_2008_2016_review.tsv", encoding = "utf8", errors = "ignore") as tsvfile:
    HIT = []
    REPORT = []
    HIT2 = []
    REPORT2 = []
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    for row in reader:
        x = randint(1,5)
        if x % 5 != 0: 
            HIT.append(row["HIT"])
            REPORT.append(row["REPORT"])
        else:
            HIT2.append(row["HIT"]) 
            REPORT2.append(row["REPORT"])

#Making the HIT sets
Y_train = np.array(HIT)
Y_test = np.array(HIT2)

#Making the Report sets
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(REPORT)
X_test_counts = count_vect.transform(REPORT2) #changed from fit transform

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

X_train_dense = X_train_tfidf.todense()
X_test_dense = X_test_tfidf.todense() #dense is for for gaussian naive bayes

#Counting training/test
print("Training samples: ", HIT.count("0")+HIT.count("1"),  "Test samples: ", HIT2.count("0")+HIT2.count("1"))

#Training model
clf = svm.LinearSVC()
model_1 = clf.fit(X_train_tfidf, Y_train)
print("SVM accuracy: " + str(model_1.score(X_test_tfidf, Y_test))) 

nb = BernoulliNB()
model_2 = nb.fit(X_train_dense, Y_train)
print("Naive Bayes accuracy: " + str(model_2.score(X_test_dense, Y_test)))

lr = linear_model.LogisticRegression()
model_3 = lr.fit(X_train_dense, Y_train)
print("Logistic Regression accuracy: " + str(model_3.score(X_test_dense, Y_test)))

rfc = RandomForestClassifier()
model_4 = rfc.fit(X_train_dense, Y_train)
print("Random Forest accuracy: " + str(model_4.score(X_test_dense, Y_test)))

#Baseline predictions
clfpredict = clf.predict(X_test_dense)
print(clfpredict, len(clfpredict))
lrpredict = lr.predict(X_test_dense)
print(lrpredict, len(lrpredict))
nbpredict = nb.predict(X_test_dense)
print(nbpredict,  len(nbpredict))
"""
#CNN predictions
cnnpred = model.predict(X_test_tfidf)
newcnnpred = []
for prediction in cnnpred:
    if prediction >= 0.5:
        newcnnpred.append("1")
    else:
        newcnnpred.append("0")
 """       
#Combining predictions
count = 0
finalpredictions = []
for prediction in clfpredict:
    y = int(prediction) + int(lrpredict[x]) + int(nbpredict[x]) #change to prediction of cnn
    if y >= 2:
        finalpredictions.append("1")
    else:
        finalpredictions.append("0")
    count +=1
"""
#1 vs 0 guess frequency compared to actual
count1 = 0
for predict in clfpredict:
    if predict == "1":
        count1 +=1

count2 = 0
for predict in lrpredict:
    if predict == "1":
        count2 +=1

count3 = 0
for predict in nbpredict:
    if predict == "1":
        count3 +=1

count4 = 0
for classification in Y_test:
        if classification == "1":
            count4 +=1
            
clfpercent = count1/len(clfpredict)
lrpercent = count2/len(lrpredict)
nbpercent = count3/len(nbpredict)
truepercent = count4/len(Y_test)

print("SVM guesses 1 "+ str(clfpercent*100) + " percent of the time")
print("Logistic regression guesses 1 "+ str(lrpercent*100) + " percent of the time")
print("Naive Bayes guesses 1 "+ str(nbpercent*100) + " percent of the time")
print("The true percentage of 1 is "+ str(truepercent*100))
"""
#Testing combined predictions
right = 0
wrong = 0
counter = 0
for prediction in finalpredictions:
    if prediction == Y_test[counter]:
        right += 1
    else:
        wrong +=1
    counter += 1
accuracy = right/(right+wrong)
print("Combined accuracy is "+ str(accuracy))

"""
try two models only - 
if disagree, always guess 1, or 50/50
"""
