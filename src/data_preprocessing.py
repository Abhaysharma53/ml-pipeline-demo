import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

def load_data(data_path):
    try:
        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
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

def data_impute(train_df, test_df):
    try:
        train_df.fillna('', inplace=True)
        test_df.fillna('', inplace=True)
        return train_df, test_df
    except Exception as e:
        print(f"An unexpected error occurred while imputing data: {e}")
        raise

def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        print(f"An unexpected error occurred during lemmatization: {e}")
        raise

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        print(f"An unexpected error occurred while removing stopwords: {e}")
        raise

def removing_numbers(text):
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        print(f"An unexpected error occurred while removing numbers: {e}")
        raise

def lower_case(text):
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        print(f"An unexpected error occurred while converting to lower case: {e}")
        raise

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        print(f"An unexpected error occurred while removing punctuations: {e}")
        raise

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        print(f"An unexpected error occurred while removing URLs: {e}")
        raise

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except KeyError:
        print("Error: 'text' column not found in the dataframe.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while removing small sentences: {e}")
        raise

def normalize_text(df):
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stopwords(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        return df
    except KeyError:
        print("Error: 'content' column not found in the dataframe.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while normalizing text: {e}")
        raise

def save_processed_data(train_df, test_df, data_path):
    try:
        data_path = os.path.join(data_path, 'processed')
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
    except IOError as e:
        print(f"Error: An I/O error occurred while saving processed data: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving processed data: {e}")
        raise

def main():
    try:
        train_df, test_df = load_data(data_path='data/raw')
        train_df, test_df = data_impute(train_df, test_df)
        train_process_data = normalize_text(train_df)
        test_process_data = normalize_text(test_df)
        save_processed_data(train_process_data, test_process_data, data_path='data')
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()
