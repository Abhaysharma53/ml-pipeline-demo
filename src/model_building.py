import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier

train_data = pd.read_csv('data/features/train_bow.csv')

X_train = train_data.iloc[:, 0:-1].values
Y_train =train_data.iloc[:, -1].values

#model initialization
clf = GradientBoostingClassifier(n_estimators = 50)
clf.fit(X_train, Y_train)

#save pickle
pickle.dump(clf, open('model.pkl', 'wb'))