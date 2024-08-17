import numpy as np
import pandas as pd
import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

def load_params(url):
    try:
        with open(url, 'r') as file:
            params = yaml.safe_load(file)
        max_feature = params.get('feature_engineering', {}).get('max_features', 1000)  # Default to 1000 if not specified
        return max_feature
    except FileNotFoundError:
        print(f"Error: The file {url} was not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except KeyError:
        print("Error: 'feature_engineering' or 'max_features' key not found in params.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading parameters: {e}")
        raise

def load_data(data_path):
    try:
        train_df = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
        train_df.fillna('', inplace=True)
        test_df.fillna('', inplace=True)
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: File not found in path {data_path}. Details: {e}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: One of the CSV files is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: Error parsing one of the CSV files.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        raise

def apply_count_vectorizer(train_df, test_df, max_feature):
    try:
        vectorizer = CountVectorizer(max_features=max_feature)
        X_train = train_df['content'].values
        Y_train = train_df['sentiment'].values

        X_test = test_df['content'].values
        Y_test = test_df['sentiment'].values
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df_bow = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
        train_df_bow['label'] = Y_train

        test_df_bow = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
        test_df_bow['label'] = Y_test
        
        return train_df_bow, test_df_bow
    except KeyError as e:
        print(f"Error: Column not found in dataframe: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while applying CountVectorizer: {e}")
        raise

def save_FE_data(train, test, data_path):
    try:
        data_path = os.path.join(data_path, 'features')
        os.makedirs(data_path, exist_ok=True)
        train.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
        test.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
    except IOError as e:
        print(f"Error: An I/O error occurred while saving feature engineering data: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving feature engineering data: {e}")
        raise

def main():
    try:
        max_feature = load_params('params.yaml')
        train_data, test_data = load_data(data_path='data/processed')
        train_FE_data, test_FE_data = apply_count_vectorizer(train_data, test_data, max_feature)
        save_FE_data(train_FE_data, test_FE_data, data_path='data')
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()
