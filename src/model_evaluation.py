import numpy as np
import pandas as pd
import os
import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def load_model(url):

    clf = pickle.load(open('model.pkl', 'rb'))
    return clf

def load_data(datapath):
    test_data = pd.read_csv(os.path.join(datapath, 'test_bow.csv'))
    return test_data

def evaluate_model(clf, X_test, Y_test):
    Y_pred = clf.predict(X_test)
    Y_pred_proba = clf.predict_proba(X_test)[:, 1]
    #compute evaluation metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred_proba)
    metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    return metrics_dict

def save_metrics_dict(metrics, file_path):
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent = 4)

def main():
    clf = load_model('model.pkl')
    test_data = load_data(datapath= 'data/features')
    X_test = test_data.iloc[:, :-1]
    Y_test = test_data.iloc[:, -1]
    metrics_dict = evaluate_model(clf, X_test, Y_test)
    save_metrics_dict(metrics_dict, 'metrics.json')

if __name__ == '__main__':
    main()


# X_test = test_data.iloc[:, 0:-1]
# Y_test = test_data.iloc[:, -1]

# Y_pred = clf.predict(X_test)
# Y_pred_proba = clf.predict_proba(X_test)[:, 1]

# # Calculate evaluation metrics
# accuracy = accuracy_score(Y_test, Y_pred)
# precision = precision_score(Y_test, Y_pred)
# recall = recall_score(Y_test, Y_pred)
# auc = roc_auc_score(Y_test, Y_pred_proba)

# metrics_dict={
#     'accuracy':accuracy,
#     'precision':precision,
#     'recall':recall,
#     'auc':auc
# }

# with open('metric.json', 'w') as file:
#     json.dump(metrics_dict, file, indent = 4)

