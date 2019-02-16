from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

digits = load_digits()
print(digits.data.shape)
print(digits.target.shape)

plt.figure(figsize = (20,4))
for index, (image, label) in enumerate(zip(digits.data[:5], digits.target[:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n'%label, fontsize=20)

plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 2)

from sklearn.neighbors import KNeighborsClassifier

logreg = KNeighborsClassifier()
logreg.fit(x_train, y_train)

print(logreg.predict(x_test[0:10]))

predictions = logreg.predict(x_test)

score = logreg.score(x_test, y_test)
from sklearn.metrics import accuracy_score
score1 = accuracy_score(predictions, y_test)
print('score: ', score)
print('score1: ', score1)


#Confusion Matric representation

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot = True, fmt='.3f', linewidth=0.5, square=True, cmap='Blues_r')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')

title = 'Accuracy Score: {0}'.format(score)

plt.title(title, size = 15)
plt.show()