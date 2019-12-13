import argparse
import os 
import re
import numpy as np
import pandas as pd
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def get_data():
    data = pd.read_csv(os.path.join(ARGS.path_data_raw, ARGS.dataset, ARGS.dataset+'.txt'), sep='\t', header=None)
    data.columns = ['review', 'target']
    return data


def clean_data(data):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    # parse through reviews in data
    clean_data = []
    for review in data:
        # remove html tags
        review = re.sub(re.compile('<.*?>'), '\n', review)
        # tokenize review into sentences
        review = tokenize.sent_tokenize(review)
        # parse through sentences in review
        clean_review = []
        for sentence in review:
            # convert to lowercase
            sentence = sentence.lower()
            # keep only alphanumeric and space characters
            sentence = re.sub(r'[^\w\s]', '', sentence)
            # remove numeric characters
            sentence = re.sub(r'[0-9]+', '', sentence)
            # remove spaces at start and end of sentences
            sentence = sentence.strip()
            # replace all whitespaces and newlines with one space character
            sentence = re.sub(r'\s+', ' ', sentence)
            # tokenize sentence into words
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(sentence)
            # remove stop words
            filtered_words = [w for w in tokens if len(w) > 2 if not w in stop_words]
            # lemmatize words
            lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
            # concatenate words into sentence
            clean_sentence = ' '.join(lemmatized_words)
            # append clean sentence
            clean_review.append(clean_sentence)
        # append clean review
        clean_data.append(clean_review)
    return clean_data
    

def encode_data(data):
    # initialize tokenizer
    tokenizer = Tokenizer()
    # create internal vocabulary based on text
    flat_data = [sentence for review in data for sentence in review]
    tokenizer.fit_on_texts(flat_data)
    # transform each sentence in review to a sequence of integers
    encoded_data = data.map(lambda s: tokenizer.texts_to_sequences(s))
    # convert dictionary to desired format
    dictionary = tokenizer.word_index
    reverse_dictionary = {value: key for key, value in dictionary.items()}
    return encoded_data, reverse_dictionary


def split_data(df):
    splits = {}
    target = pd.DataFrame(data = np.asarray(df['target']), columns=['target'])
    data = pd.DataFrame(data = np.asarray(df['codes']), columns=['codes'])
    data['numerics'] = [[[0 for i in range(0,1)] for i in range(0,1)] for i in range(df.shape[0])]
    data['to_event'] = np.zeros(df.shape[0])
    splits['data_train'], splits['data_test'], splits['target_train'], splits['target_test'] = train_test_split(data, target, test_size=0.3, random_state=7)
    return splits


def pickle_data(splits):
    for split in splits.keys():
        pd.to_pickle(splits[split], os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, split+'.pkl'))
        print()
        print('data frame for "' + split + '":')
        print()
        print(splits[split].head())
    print()


def pickle_dictionary(dictionary):
    pd.to_pickle(dictionary, os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, 'dictionary.pkl'))


def main(ARGS):
    '''Main body of the code'''
    # get dataset as a data frame
    df = get_data()
    print()
    print('example review before any preprocessing:')
    print()
    print(df['review'][3])
    # clean data
    df['clean'] = clean_data(df['review'])
    print()
    print('example review after cleaning:')
    print()
    print(df['clean'][3])
    # encode data
    df['codes'], dictionary = encode_data(df['clean'])
    print()
    print('example review after encoding:')
    print()
    print(df['codes'][3])
    print()
    print('dictionary size: ' + str(len(dictionary)))
    print()
    dictionary_head = {k: dictionary[k] for k in list(dictionary)[:10]}
    print('first ten examples from "dictionary":')
    print()
    print(dictionary_head)
    print()
    print('data frame size: ' + str(df.shape[0]))
    print()
    print('data frame before splitting between "data" and "target":')
    print()
    print(df.head())
    print()
    # split data into desired format
    df_splits = split_data(df)
    # pickle data and dictionary
    pickle_data(df_splits)
    pickle_dictionary(dictionary)


def parse_arguments(parser):
    '''Read user arguments'''
    parser.add_argument('--path_data_raw', 
                        type=str,  
                        default='../data',  
                        help='Path to raw data')
    parser.add_argument('--path_data_preprocessed', 
                        type=str,  
                        default='data',  
                        help='Path to preprocessed data')
    parser.add_argument('--dataset',  
                        type=str,  
                        choices=['IMDB', 'yelp'],  
                        default='IMDB',  
                        help='Select dataset')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)