import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn import metrics

digits = load_digits()
print('data: ', digits.data.shape)
print('labels: ', digits.target.shape)
print(type(digits))

plt.figure(figsize=(20,4))
'''
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
	plt.subplot(1, 5, index + 1)
	plt.imshow(np.reshape(image, (8,8)), cmap = plt.cm.gray)
	plt.title('Training: %i\n' %label, fontsize = 20)
	plt.show()
'''

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state = 2)

print(x_train.shape)
print(y_train.shape)

logReg = LogisticRegression()

logReg.fit(x_train, y_train)

'''
plt.figure(figsize=(4,4))
plt.imshow(np.reshape(x_test[0], (8,8)), cmap = plt.cm.gray)
'''
print(y_test[0])
print(logReg.predict(x_test[0].reshape(1,-1)))

predictions = logReg.predict(x_test)
score = logReg.score(x_test, y_test)
print('score: ', score)

cm = metrics.confusion_matrix(y_test, predictions)

print('confusion matrix \n',cm)