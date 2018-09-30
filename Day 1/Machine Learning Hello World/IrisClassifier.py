import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
irisdata = pd.read_csv('iris.csv', skiprows = 1, usecols=[0,1,2,3], names = ['Sepal_len', 'Sepal_wid', 'Petal_len', 'Petal_wid'])

labels_df = pd.read_csv('iris.csv', header = None, skiprows=1, usecols=[4], names=['labels'])

features = irisdata.values
labels = labels_df.values.ravel()

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 2)

logReg = LogisticRegression()

print(x_train.shape)
print(y_train.shape)
#print(x_train)
#print(y_train)

logReg.fit(x_train, y_train)
predictions = logReg.predict(x_test)

score = logReg.score(x_test, y_test) 
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
print('predicted\tactual')
for predicted, actual in zip(predictions, y_test):
    print(predicted, actual, sep='\t')