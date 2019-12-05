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
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def combine_files():
    with open(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'.txt'), 'w') as outfile:
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'-'+split+'.txt')) as infile:
                outfile.write(infile.read())


def get_vocab_size():
    with open(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'.txt')) as f:
        c = collections.Counter(f.read().split())
    return len(c.keys())


def get_data():
    data = pd.read_csv(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'.txt'), sep='\t', header=None)
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
    dictionary1 = tokenizer.word_index
    dictionary2 = [[value, key] for key, value in dictionary1.iteritems()]
    dictionary = pd.DataFrame(data=dictionary2)
    return encoded_data, tokenizer, dictionary


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
        # TODO: pickle once in desired format
        pd.to_pickle(splits[split], os.path.join(DATA_PATH, split+'.pkl'))
        # TODO: remove print statements later
        print()
        print('data frame for "' + split + '":')
        print()
        print(splits[split].head())
    print()


def pickle_dictionary(dictionary):
    pd.to_pickle(dictionary, "dictionary.pkl")


if __name__ == '__main__':

    # define parser to parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='data', help='define path to data')
    parser.add_argument('--dataset', type=str, choices=['IMDB', 'yelp'], default='IMDB', help='select dataset')
    parser.add_argument('--encoding', type=str, choices=['one-hot', 'word-to-vec'], default='one-hot', help='select encoding method')
    args = parser.parse_args()

    # define constants
    DATA_PATH = args.datapath
    DATA_SET = args.dataset
    DATA_SPLITS = ['train', 'test']

    # combine original dataset files combined into one
    # combine_files() # TODO: comment out once this is done

    # get number of unique words in dataset
    vocab_size = get_vocab_size()

    # get dataset as a data frame
    df = get_data()

    # TODO: remove print statements later
    print()
    print('example review before cleaning:')
    print()
    print(df['review'][3])

    # clean data
    df['clean'] = clean_data(df['review'])

    # TODO: remove print statements later
    print()
    print('example review after cleaning:')
    print()
    print(df['clean'][3])

    # encode data
    df['codes'], tokenizer, dictionary = encode_data(df['clean'])

    # TODO: remove print statements later
    print()
    print('example review after encoding:')
    print()
    print(df['codes'][3])
    print()
    print('word_index:')
    print()
    print(tokenizer.word_index)

    # TODO: remove print statements later
    print()
    print('data fram size: ' + str(df.shape[0]))
    print()
    print('data frame before splitting between "data" and "target":')
    print()
    print(df.head())
    print()

    # split data into format required for RETAIN
    df_splits = split_data(df)

    # pickle data and dictionary
    pickle_data(df_splits)
    pickle_dictionary(dictionary)