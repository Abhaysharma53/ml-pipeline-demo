import numpy as np
import pandas as pd
import pickle
import yaml

from sklearn.ensemble import GradientBoostingClassifier

params = yaml.safe_load(open('params.yaml', 'r'))
n_estimator = params['model_building']['n_estimators']
learn_rate = params['model_building']['learning_rate']
train_data = pd.read_csv('data/features/train_bow.csv')

X_train = train_data.iloc[:, 0:-1]
Y_train =train_data.iloc[:, -1]

#model initialization
clf = GradientBoostingClassifier(n_estimators  = n_estimator, learning_rate= learn_rate)
clf.fit(X_train, Y_train)

#save pickle
pickle.dump(clf, open('model.pkl', 'wb'))