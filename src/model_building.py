import numpy as np
import pandas as pd
import pickle
import yaml
import os

from sklearn.ensemble import GradientBoostingClassifier

def load_params(url):
    try:
        with open(url, 'r') as file:
            params = yaml.safe_load(file)
        n_estimator = params.get('model_building', {}).get('n_estimators', 100)  # Default to 100 if not specified
        learn_rate = params.get('model_building', {}).get('learning_rate', 0.1)  # Default to 0.1 if not specified
        return n_estimator, learn_rate
    except FileNotFoundError:
        print(f"Error: The file {url} was not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except KeyError:
        print("Error: 'model_building', 'n_estimators', or 'learning_rate' key not found in params.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading parameters: {e}")
        raise

def load_feature(datapath):
    try:
        train_data = pd.read_csv(os.path.join(datapath, 'train_bow.csv'))
        return train_data
    except FileNotFoundError as e:
        print(f"Error: File not found in path {datapath}. Details: {e}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: Error parsing the CSV file.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading features: {e}")
        raise

def train_model(X_train, Y_train, n_estimator, learn_rate):
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimator, learning_rate=learn_rate)
        clf.fit(X_train, Y_train)
        return clf
    except ValueError as e:
        print(f"Error: Value error during model training. Details: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while training the model: {e}")
        raise

def save_model(model):
    try:
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
    except IOError as e:
        print(f"Error: An I/O error occurred while saving the model: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving the model: {e}")
        raise

def main():
    try:
        n_estimator, learn_rate = load_params('params.yaml')
        train_data = load_feature(datapath='data/features')
        X_train = train_data.iloc[:, :-1]
        Y_train = train_data.iloc[:, -1]
        clf = train_model(X_train, Y_train, n_estimator=n_estimator, learn_rate=learn_rate)
        save_model(clf)
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()
