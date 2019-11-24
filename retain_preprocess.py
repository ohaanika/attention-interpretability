import numpy as np
import pandas as pd
import re
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

# define constants
FOLDER_NAME = 'data/'
FILE_NAMES = ['-train', '-valid', '-test']

def combine_files(dataset):
    with open(FOLDER_NAME + dataset + '.txt', 'w') as outfile:
        for file_name in FILE_NAMES:
            with open(FOLDER_NAME + dataset + file_name + '.txt') as infile:
                outfile.write(infile.read())

def get_vocab_size(dataset):
    with open(FOLDER_NAME + dataset + '.txt') as f:
        c = collections.Counter(f.read().split())
    return len(c.keys())

def get_data(dataset):
    data = pd.read_csv(FOLDER_NAME + 'IMDB.txt', sep='\t', header = None)
    data.columns = ['sentence', 'target']
    return data

def process_data(data):
    # lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()
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
        # EXTRA STUFF
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        # append to new list of cleaned sentences 
        clean_data.append(tokens)
    return (clean_data)
    
    # def preprocess1(sentence):
    #     sentence = tokenize.sent_tokenize(sentence)
    #     return sentence

    # def preprocess2(sentence):
    #     tokenizer = RegexpTokenizer(r'\w+')
    #     tokens = tokenizer.tokenize(rem_num)  
    #     filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    #     stem_words=[stemmer.stem(w) for w in filtered_words]
    #     lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    #     return ' '.join(lemma_words)

if __name__ == '__main__':
    dataset = 'IMDB'
    # combine_files(dataset)
    vocab_size = get_vocab_size(dataset)
    data = get_data(dataset)
    data['clean'] = process_data(data['sentence'])
    print('----- ORIGINAL -----')
    print(data['sentence'][3])
    print()
    print('----- CLEAN -----')
    print(data['clean'][3])
    # print(data.head())