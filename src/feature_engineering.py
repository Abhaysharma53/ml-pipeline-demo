import numpy as np
import pandas as pd
import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

def load_params(url):
    params = yaml.safe_load(open('params.yaml', 'r'))
    max_feature = params['feature_engineering']['max_features']
    return max_feature

def load_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)
    return train_df, test_df

#processing to apply BOW
def apply_count_vectorizer(train_df, test_df, max_feature):
    vectorizer = CountVectorizer(max_features= max_feature)
    X_train = train_df['content'].values
    Y_train = train_df['sentiment'].values

    X_test = test_df['content'].values
    Y_test = test_df['sentiment'].values
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = Y_train

    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = Y_test
    return train_df, test_df

def save_FE_data(train, test, data_path):
    data_path = os.path.join(data_path, 'features')
    os.makedirs(data_path)
    train.to_csv(os.path.join(data_path, 'train_bow.csv'), index = False)
    test.to_csv(os.path.join(data_path, 'test_bow.csv'), index = False)

# train_data.fillna('', inplace = True)
# test_data.fillna('', inplace = True)
def main():
    max_feature = load_params('params.yaml')
    train_data, test_data = load_data(data_path= 'data/processed')
    train_FE_data, test_FE_data = apply_count_vectorizer(train_data, test_data, max_feature)
    save_FE_data(train_FE_data, test_FE_data, data_path= 'data')

if __name__ == '__main__':
    main()
    
  
    







# train_df = pd.DataFrame(X_train_bow.toarray())
# train_df['label'] = Y_train

# test_df = pd.DataFrame(X_test_bow.toarray())
# test_df['label'] = Y_test

# data_path = os.path.join('data', 'features')
# os.makedirs(data_path)

# train_df.to_csv(os.path.join(data_path, 'train_bow.csv'))

# test_df.to_csv(os.path.join(data_path, 'test_bow.csv'))






