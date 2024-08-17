import numpy as np
import pandas as pd
import os
import yaml

from sklearn.model_selection import train_test_split

def load_params(url: str) -> dict:
    try:
        with open(url, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params.get('data_ingestion', {}).get('test_size', 0.2)  # Default to 0.2 if not specified
        return test_size
    except FileNotFoundError:
        print(f"Error: The file {url} was not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except KeyError:
        print("Error: 'data_ingestion' or 'test_size' key not found in params.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading parameters: {e}")
        raise

def load_dataset(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.EmptyDataError:
        print("Error: The data file is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: Error parsing the data file.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
    except KeyError:
        print("Error: 'tweet_id' column not found in the dataframe.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while dropping columns: {e}")
        raise
    
    final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
    
    if final_df.empty:
        print("Warning: No data left after filtering sentiments.")
    
    try:
        final_df['sentiment'].replace({'neutral': 0, 'sadness': 1}, inplace=True)
    except KeyError:
        print("Error: 'sentiment' column not found in the dataframe.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while replacing sentiment values: {e}")
        raise
    
    return final_df

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)  # `exist_ok=True` prevents error if directory already exists
        train_df.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    except IOError as e:
        print(f"Error: An I/O error occurred while saving data: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving data: {e}")
        raise

def main():
    try:
        test_size = load_params(url='params.yaml')
        df = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='data')
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main() 