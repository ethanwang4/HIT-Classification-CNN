#Imports for baseline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm
from sklearn import linear_model
from random import randint
import numpy as np
import csv


#Imports for CNN
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.layers import Activation, Dense, Dropout
from keras.layers import Input, Concatenate
from keras.models import Model
from keras.models import Sequential
from keras import regularizers 
import keras_metrics


#More imports
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
import pandas as pd
import importlib
import pickle
import os


#Initializing CNN variables
base_dir = ''
glove_dir = os.path.join(base_dir,"")
max_num_words = 20000 
max_sequence_length = 1000
embedding_dim = 50
batch_size = 50
num_epochs = 6
print("Building a Convolutional Neural Network:")


#Loading and parsing pre-trained embedding vectors
embeddings_index = {} 
f = open(os.path.join(glove_dir, '/Users/ethanwang/Desktop/CS stuff/Summer Project/glove.6B/glove.6B.50d.txt'), encoding = "utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors' % len(embeddings_index))  


#Loading and parsing MAUDE reports
with open("MAUDE_2008_2016_review.tsv", encoding = "utf8", errors = "ignore") as tsvfile:
    HIT = []
    REPORT = []
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    for row in reader:
        HIT.append(row["HIT"])
        REPORT.append(row["REPORT"])


#Tokenization and Labeling
tokenizer = Tokenizer(num_words = max_num_words)
tokenizer.fit_on_texts(REPORT)
sequences = tokenizer.texts_to_sequences(REPORT) 
padded = pad_sequences(sequences, maxlen=max_sequence_length)

count_vect = CountVectorizer() 
X_counts = count_vect.fit_transform(REPORT)
tfidf_transformer = TfidfTransformer() 
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_dense = X_tfidf.todense()

x_train = [] 
x_val = []
x_test = []

x_train_base = [] 
x_val_base = []
x_test_base = []

y_train = []
y_val = []
y_test = []

num = 0
for entry in padded:
    x = randint(1,10)
    if x >8:
        x_test.append(entry)
        x_test_base.append(X_dense[num])
        y_test.append(HIT[num])
    elif x >7:
        x_val.append(entry)
        x_val_base.append(X_dense[num])
        y_val.append(HIT[num])
    else:
        x_train.append(entry)
        x_train_base.append(X_dense[num])
        y_train.append(HIT[num])
    num += 1

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)

x_train_base = np.array(x_train_base)
x_val_base = np.array(x_val_base)
x_test_base = np.array(x_test_base)

x_train_base = x_train_base.reshape((len(y_train),17212)) 
x_val_base = x_val_base.reshape((len(y_val),17212))
x_test_base = x_test_base.reshape((len(y_test),17212))

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


#Encoding/labeling HIT sets 
encoder = LabelBinarizer()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)


#Forming embedding matrix
print('Preparing embedding matrix')
num_words = min(max_num_words, len(word_index) + 1)
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        """ words not found in embedding index will be all-zeros."""
        embedding_matrix[i] = embedding_vector


#Building CNN model - Improved
convs = []
kernel_size = [2, 3, 4]

embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=True)

sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

convs = []
for ks in kernel_size:
        l_conv = Conv1D(256, ks, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(int(max_sequence_length-ks+1))(l_conv)
        l_pool = Dropout(0.3)(l_pool)
        convs.append(l_pool)

l_merge = Concatenate(axis = 1)(convs)
l_flat = Flatten()(l_merge)
l_dense = Dense(1024, activation='relu')(l_flat)
l_dense = Dropout(0.5)(l_dense)
preds = Dense(1, activation='sigmoid')(l_dense)

model = Model(sequence_input, preds)


#CNN model summary and compilation 
print("Model summary:")
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer="adam",
              metrics=['accuracy', keras_metrics.precision(),
                       keras_metrics.recall(), keras_metrics.true_positive()])


#Creating a callback
es = EarlyStopping(
    monitor = "val_loss",
    min_delta = 0.01,
    patience = 1,
    verbose = 1,
    mode = "max")


#Training and testing CNN model 
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose= 2,
          validation_data=(x_val,y_val))
score = model.evaluate(x_test, y_test, verbose = 2)
print("CNN loss = ", score[0])
print("CNN accuracy = ", score[1])


#Training baseline models - change it to 70 instead of 80 
print(" ")
print("Building Baseline Models:")

clf = svm.LinearSVC()
model_1 = clf.fit(x_train_base, y_train.ravel())
print("SVM accuracy: " + str(model_1.score(x_test_base, y_test))) 

nb = BernoulliNB()
model_2 = nb.fit(x_train_base, y_train.ravel())
print("Naive Bayes accuracy: " + str(model_2.score(x_test_base, y_test)))

lr = linear_model.LogisticRegression()
model_3 = lr.fit(x_train_base, y_train.ravel())
print("Logistic Regression accuracy: " + str(model_3.score(x_test_base, y_test)))

rfc = RandomForestClassifier()
model_4 = rfc.fit(x_train_base, y_train.ravel())
print("Random Forest accuracy: " + str(model_4.score(x_test_base, y_test)))


#Baseline predictions + probability predictions 
clfpredict = clf.predict(x_test_base)
lrpredict = lr.predict(x_test_base)

clf2 = CalibratedClassifierCV(clf)
clf2.fit(x_train_base, y_train.ravel()) 

clfprob = clf2.predict_proba(x_test_base) 
clfprob2 = []
for x in clfprob:
    clfprob2.append(x[1])
lrprob = lr.predict_proba(x_test_base) 
lrprob2 = []
for x in lrprob:
    lrprob2.append(x[1])

nbpredict = nb.predict(x_test_base)
rfcpredict = rfc.predict(x_test_base)


#CNN predictions + adjustment to 0 or 1 
cnnpred = model.predict(x_test)
newcnnpred = []
for x in cnnpred:
    if x >= 0.5:  
        newcnnpred.append("1")
    else:
        newcnnpred.append("0")


#Combining predictions
count = 0
finalpredictions = []
for x in clfprob2:
    y = x + lrprob2[count] + cnnpred[count] #+ x or lrpob2 change to cnn 
    if y >= 1.5: #change to 1
        finalpredictions.append("1")
    else:
        finalpredictions.append("0")
    count +=1

z = 0
finalpredictions2 = [] 
for x in clfprob2:
    y = x + lrprob2[z] 
    if y >= 1:  
        finalpredictions2.append("1")
    else:
        finalpredictions2.append("0")
    z +=1

zee = 0
finalpredictions3 = []
for x in clfprob2:
    y = x + cnnpred[zee] 
    if y >= 1:  
        finalpredictions3.append("1")
    else:
        finalpredictions3.append("0")
    zee +=1

zeee = 0
finalpredictions4 = []
for x in clfprob2:
    y = lrprob2[zeee]+ cnnpred[zeee] 
    if y >= 1:  
        finalpredictions4.append("1")
    else:
        finalpredictions4.append("0")
    zeee +=1


#Testing combined predictions
finalpredictions = list(map(int, finalpredictions))
finalpredictions2 = list(map(int, finalpredictions2)) 
finalpredictions3 = list(map(int, finalpredictions3))
finalpredictions4 = list(map(int, finalpredictions4))

acc = accuracy_score(y_test, finalpredictions)
print(" ")
print("Combined Model:")
print("Combined accuracy of CNN, SVM, LR is " + str(acc))

acc2 = accuracy_score(y_test, finalpredictions2)
print("Combined accuracy of SVM, LR is " + str(acc2))

acc3 = accuracy_score(y_test, finalpredictions3)
print("Combined accuracy of CNN, SVM is " + str(acc3))

acc4 = accuracy_score(y_test, finalpredictions4)
print("Combined accuracy of CNN, LR is " + str(acc4))



#Precision and recall scores
print(" ")
print("Precision, Recall, and F1 Metrics:")
newcnnpred = list(map(int, newcnnpred))
lrpredict = list(map(int, lrpredict))
clfpredict = list(map(int, clfpredict))
nbpredict = list(map(int, nbpredict))
rfcpredict = list(map(int, rfcpredict))

print("CNN precision = ", precision_score(y_test, newcnnpred))
print("Logistic regression precision = ", precision_score(y_test, lrpredict))
print("SVM precision = ", precision_score(y_test, clfpredict))
print("Naive Bayes precision = ", precision_score(y_test, nbpredict))
print("Random Forest precision = ", precision_score(y_test, rfcpredict))
print("Combined - CNN/SVM/LR precision = ", precision_score(y_test, finalpredictions))
print("Combined - SVM/LR precision = ", precision_score(y_test, finalpredictions2))
print("Combined - CNN/SVM precision = ", precision_score(y_test, finalpredictions3))
print("Combined - CNN/LR precision = ", precision_score(y_test, finalpredictions4))

print("CNN recall = ", recall_score(y_test, newcnnpred))
print("Logistic regression recall = ", recall_score(y_test, lrpredict))
print("SVM recall = ", recall_score(y_test, clfpredict))
print("Naive Bayes recall = ", recall_score(y_test, nbpredict))
print("Random Forest recall = ", recall_score(y_test, rfcpredict))
print("Combined - CNN/SVM/LR recall = ", recall_score(y_test, finalpredictions))
print("Combined - SVM/LR recall = ", recall_score(y_test, finalpredictions2))
print("Combined - CNN/SVM recall = ", recall_score(y_test, finalpredictions3))
print("Combined - CNN/LR recall = ", recall_score(y_test, finalpredictions4))

print("CNN f1 score = ", f1_score(y_test, newcnnpred))
print("Logistic regression f1 score = ", f1_score(y_test, lrpredict))
print("SVM f1 score = ", f1_score(y_test, clfpredict))
print("Naive Bayes f1 score = ", f1_score(y_test, nbpredict))
print("Random Forest f1 score = ", f1_score(y_test, rfcpredict))
print("Combined - CNN/SVM/LR f1 score = ", f1_score(y_test, finalpredictions))
print("Combined - SVM/LR f1 score = ", f1_score(y_test, finalpredictions2))
print("Combined - CNN/SVM f1 score = ", f1_score(y_test, finalpredictions3))
print("Combined - CNN/LR f1 score = ", f1_score(y_test, finalpredictions4))


"""
Notes: Increasing dropout improves CNN precision, but worsens recall
CNN accuracy = around 85-86
"""
