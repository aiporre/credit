import pandas as pd
from sklearn import svm, preprocessing, cross_validation
import time
import matplotlib.pyplot as plt
import numpy as np


credit = pd.read_csv('data/training.csv')
# credit.set_index('Unnamed', inplace=True)
# credit.plot()
# plt.tit# plt.show()
print credit.head()
# X = np.array(credit.iloc[:,:-1])
# X = preprocessing.scale(X)
# y = np.array(credit.iloc[:,-1])
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
# classifier = svm.SVC(kernel='rbf    ',verbose=True)
# now = time.time()
# print 'Training.....'
# classifier.fit(X_train, y_train)
# print 'end of training before ' + str(time.time() - now) + ' seconds'
# print 'Classification score: ' + str(classifier.score(X_test, y_test))
# print ''




