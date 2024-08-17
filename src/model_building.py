import numpy as np
import pandas as pd
import pickle
import yaml
import os

from sklearn.ensemble import GradientBoostingClassifier

def load_params(url):
    params = yaml.safe_load(open('params.yaml', 'r'))
    n_estimator = params['model_building']['n_estimators']
    learn_rate = params['model_building']['learning_rate']
    return n_estimator, learn_rate

def load_feature(datapath):
    train_data = pd.read_csv(os.path.join(datapath, 'train_bow.csv'))
    return train_data

def train_model(X_train, Y_train, n_estimator, learn_rate):
    clf = GradientBoostingClassifier(n_estimators  = n_estimator, learning_rate= learn_rate)
    clf.fit(X_train, Y_train)
    return clf

def save_model(model):
    pickle.dump(model, open('model.pkl', 'wb'))

def main():
    n_estimator, learn_rate = load_params('params.yaml')
    train_data = load_feature(datapath= 'data/features')
    X_train = train_data.iloc[:, :-1]
    Y_train = train_data.iloc[:, -1]
    clf = train_model(X_train, Y_train, n_estimator= n_estimator, learn_rate= learn_rate)
    save_model(clf)

if __name__ == '__main__':
    main()
    

#model initialization
# clf = GradientBoostingClassifier(n_estimators  = n_estimator, learning_rate= learn_rate)
# clf.fit(X_train, Y_train)

# #save pickle
# pickle.dump(clf, open('model.pkl', 'wb'))