#Fetching Data
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

x, y = mnist['data'], mnist['target']

x.shape

y.shape

%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

some_digit = x.loc[2500]
some_digit_image = some_digit.values.reshape(28, 28)

plt.imshow(some_digit_image, cmap='binary', interpolation='nearest')
plt.axis('off')
plt.show()

y[2500]

x_train = x[0:6000]

x_test = x[6000:7000]

y_train = y[0:6000]

y_test = y[6000:7000]

import numpy as np
shuffle_index = np.random.permutation(x_train.shape[0])
x_train = x_train.iloc[shuffle_index]
y_train = y_train.iloc[shuffle_index]

#Creating a 3 detector

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

y_train_3 = (y_train==3)
y_test_3 = (y_test==3)

y_train

#logistic regression sklearn
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(tol = 0.1)

clf.fit(x_train, y_train_3)

clf.predict([some_digit])

#cross validation

from sklearn.model_selection import cross_val_score
a = cross_val_score(clf, x_train, y_train_3, cv=3, scoring="accuracy")

a.mean()
