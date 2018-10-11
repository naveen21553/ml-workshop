import pandas as pd

features_df = pd.read_csv('iris.csv', usecols = [0,1,2,3], skiprows = 1, names = ['Sepal_Len', 'Sepal_Wid', 'Petal_Len', 'Petal_Wid'])
labels_df = pd.read_csv('iris.csv', usecols = [4], skiprows = 1, names = ['Labels'])

features = features_df.values
labels = labels_df.values.ravel()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.23, random_state = 2)

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()
logReg.fit(x_train, y_train)

print(logReg.predict(x_test[0].reshape(1, -1)))
print(y_test[0])

predictions = logReg.predict(x_test)

score = logReg.score(x_test, y_test)

print(score)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
print(cm)