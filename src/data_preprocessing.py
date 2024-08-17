import numpy as np
import pandas as pd
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
#extract data from data/raw
def load_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    return train_df, test_df

def data_impute(train_df, test_df):
    train_df.fillna('', inplace = True)
    test_df.fillna('', inplace = True)
    return train_df, test_df

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
     #remove extra whitespaces
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# def remove_small_sentence(df):
#     for i in range(len(df)):
#         if len(df.text.iloc[i].split()) < 3:
#             df.text.iloc[i] = np.nan

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content = df.content.apply(lambda content : lower_case(content))
    df.content = df.content.apply(lambda content : remove_stopwords(content))
    df.content = df.content.apply(lambda content : removing_numbers(content))
    df.content = df.content.apply(lambda content : removing_punctuations(content))
    df.content = df.content.apply(lambda content : removing_urls(content))
    #df.content = df.content.apply(lambda content : remove_small_sentences(content))
    df.content = df.content.apply(lambda content : lemmatization(content))
    return df

def save_processed_data(train_df, test_df, data_path):
    data_path = os.path.join(data_path, 'processed')
    os.makedirs(data_path)
    train_df.to_csv(os.path.join(data_path, 'train_processed.csv'))
    test_df.to_csv(os.path.join(data_path, 'test_processed.csv'))


#processing raw train & test data
def main():
    train_df, test_df = load_data(data_path= 'data/raw')
    train_df, test_df = data_impute(train_df, test_df)
    train_process_data = normalize_text(train_df)
    test_process_data = normalize_text(test_df)
    save_processed_data(train_process_data, test_process_data, data_path= 'data')

if __name__ == '__main__':
    main()






# #store the data inside data/processed
# data_path  = os.path.join("data", "processed")
# os.makedirs(data_path)

# train_process_data.to_csv(os.path.join(data_path, "train_processed.csv"))
# test_process_data.to_csv(os.path.join(data_path, "test_processed.csv"))