import argparse
import os 
import re
import numpy as np
import pandas as pd
import spacy
import nltk
import collections
from sklearn.model_selection import train_test_split
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def combine_files():
    with open(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'.txt'), 'w') as outfile:
        for split in DATA_SPLITS:
            with open(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'-'+split+'.txt')) as infile:
                outfile.write(infile.read())

def get_vocab_size():
    with open(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'.txt')) as f:
        c = collections.Counter(f.read().split())
    return len(c.keys())

def get_data():
    data = pd.read_csv(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'.txt'), sep='\t', header=None)
    data.columns = ['sentence', 'target']
    return data

def process_data(data):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    clean_data = []
    for sentence in data:
        # remove html tags
        sentence = re.sub(re.compile('<.*?>'), ' ', sentence)
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
        # append to new list of cleaned sentences 
        clean_data.append(' '.join(lemmatized_words))
    return (clean_data)

if __name__ == '__main__':

    # define parser to parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='data', help='define path to data')
    parser.add_argument('--dataset', type=str, choices=['IMDB', 'yelp'], required=True, help='select dataset')
    args = parser.parse_args()

    # define constants
    DATA_PATH = args.datapath
    DATA_SET = args.dataset
    DATA_SPLITS = ['train', 'valid', 'test']

    # prepare data
    combine_files() # TODO: comment out once dataset files combined into one
    vocab_size = get_vocab_size()
    data = get_data()
    data['clean'] = process_data(data['sentence'])

    # TODO: remove print statements once testing no longer needed
    print()
    print('----- EXAMPLE SENTENCE (ORIGINAL) -----')
    print()
    print(data['sentence'][3])
    print()
    print('----- EXAMPLE SENTENCE (CLEAN) -----')
    print()
    print(data['clean'][3])
    print()
    print('----- DATA FRAME -----')
    print(data.head())
    print()
    print('DATA FRAME SIZE: ' + str(data.shape[0]))
    print()