import numpy as np
import pandas as pd
import os
import yaml

from sklearn.model_selection import train_test_split

def load_params(url):
    params = yaml.safe_load(open('params.yaml', 'r'))
    test_size = params['data_ingestion']['test_size']
    return test_size

def load_dataset(data_url):
    df = pd.read_csv(data_url)
    return df

def preprocess_data(df):

    df.drop(columns = ['tweet_id'], inplace = True)

    final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]

    final_df['sentiment'].replace({'neutral':0, 'sadness':1}, inplace = True)
    return final_df

def save_data(train_df, test_df, data_path):
    data_path = os.path.join(data_path, 'raw')
    os.makedirs(data_path)
    train_df.to_csv(os.path.join(data_path, 'train.csv'))
    test_df.to_csv(os.path.join(data_path, 'test.csv'))

def main():
    test_size = load_params(url= 'parms.yaml')
    df = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = preprocess_data(df)
    train_data, test_data = train_test_split(final_df, test_size= test_size, random_state= 42)
    save_data(train_data, test_data, data_path= 'data')

if __name__ == '__main__':
    main()





