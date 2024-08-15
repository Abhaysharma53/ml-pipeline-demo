import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

clf = pickle.load(open('model.pkl', 'rb'))

test_data = pd.read_csv('data/features/test_bow.csv')

X_test = test_data.iloc[:, 0:-1]
Y_test = test_data.iloc[:, -1]

Y_pred = clf.predict(X_test)
Y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
auc = roc_auc_score(Y_test, Y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

with open('metric.json', 'w') as file:
    json.dump(metrics_dict, file, indent = 4)

