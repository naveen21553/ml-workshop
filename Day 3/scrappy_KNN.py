import random
from scipy.spatial.distance import euclidean as dist

class myKNN():
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
    
    def predict(self, x_test):
        predictions = []
        for item in x_test:
            prediction = self.closest(item)
            predictions.append(prediction)        
        return predictions

    def closest(self, item):
        best_dist = dist(self.X_train[0], item)
        best_index = 0

        for i in range(1, len(self.X_train)):
            distance = dist(self.X_train[i], item)
            if distance < best_dist:
                best_dist = distance
                best_index = i
        return self.Y_train[best_index]

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.23)

#from sklearn.neighbors import KNeighborsClassifier
clf = myKNN()

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(predictions, y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(predictions, y_test)

print('Score: ', score)
print(cm)