from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from sklerarn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random
from scipy.spatial.distance import euclidean

class ScrappyNN():
	def fit(self, x_train, y_train):
		self.X_train = x_train
		self.Y_train = y_train
		
	def predict(self, x_test):
		predictions = []
		for i in x_test:
			label = self.closest(i)
			predictions.append(label)
			
		return predictions
		
	def closest(self, i):
		best_dist = euclidean(i, self.X_train[0])
		best_index = 0
		
		for j in range(1, len(self.X_train)):
			dist = euclidean(i, self.X_train[j])
			if dist < best_dist:
				best_dist = dist
				best_index = j
				
		return self.Y_train[best_index]
		
iris = load_iris()
print(iris.feature_names, iris.target_names, sep='\n')

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.30, random_state = 3)

clf = ScrappyNN()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

score = accuracy_score(predictions, y_test)

print(score)

cm = confusion_matrix(y_test, predictions)

print(cm)

