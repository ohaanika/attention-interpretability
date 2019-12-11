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
from tqdm import tqdm


def get_data():
    data = pd.read_csv(os.path.join(ARGS.path_data_raw, ARGS.dataset, ARGS.dataset+'.txt'), sep='\t', header=None)
    data.columns = ['review', 'target']
    return data


def clean_data(data, p1, p2):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer =  PorterStemmer()
    # parse through reviews in data
    clean_data = []
    for review in tqdm(data):
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
            
            if p1 == "Lemmatized":
                #lemmatizer = nltk.stem.WordNetLemmatizer()
                if p2 == "NoStopWords":
                    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
                else:
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
            elif p1 == "Stemming":
                #stemmer = nltk.stem.PorterStemmer()
                if p2 == "NoStopWords":
                    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
                else:
                    tokens = [stemmer.stem(t) for t in tokens]
            # remove stop words
            #filtered_words = [w for w in tokens if len(w) > 2 if not w in stop_words]
            # lemmatize words
            #lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
            # concatenate words into sentence
            clean_sentence = ' '.join(tokens)
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


def split_data(df, p1, p2):
    splits = {}
    target = pd.DataFrame(data = np.asarray(df['target']), columns=['target'])
    if p1 == "Stemmed" and p2 =="NoStopWords":
        data = pd.DataFrame(data = np.asarray(df['codes_for_Stemmed_and_NoStopWords']), columns=['codes'])
        data['numerics'] = [[[0 for i in range(0,1)] for i in range(0,1)] for i in range(df.shape[0])]
        data['to_event'] = np.zeros(df.shape[0])
        splits['data_train'], splits['data_test'], splits['target_train'], splits['target_test'] = train_test_split(data, target, test_size=0.3, random_state=7)
    elif p1 == "Lemmatized" and p2 =="NoStopWords":
        data = pd.DataFrame(data = np.asarray(df['codes_for_Lemmatized_and_NoStopWords']), columns=['codes'])
        data['numerics'] = [[[0 for i in range(0,1)] for i in range(0,1)] for i in range(df.shape[0])]
        data['to_event'] = np.zeros(df.shape[0])
        splits['data_train'], splits['data_test'], splits['target_train'], splits['target_test'] = train_test_split(data, target, test_size=0.3, random_state=7)
    elif p1 == "Stemmed" and p2 =="":
        data = pd.DataFrame(data = np.asarray(df['codes_for_Stemmed_and_StopWords']), columns=['codes'])
        data['numerics'] = [[[0 for i in range(0,1)] for i in range(0,1)] for i in range(df.shape[0])]
        data['to_event'] = np.zeros(df.shape[0])
        splits['data_train'], splits['data_test'], splits['target_train'], splits['target_test'] = train_test_split(data, target, test_size=0.3, random_state=7)
    elif p1 == "Lemmatized" and p2 =="":
        data = pd.DataFrame(data = np.asarray(df['codes_for_Lemmatized_and_StopWords']), columns=['codes'])
        data['numerics'] = [[[0 for i in range(0,1)] for i in range(0,1)] for i in range(df.shape[0])]
        data['to_event'] = np.zeros(df.shape[0])
        splits['data_train'], splits['data_test'], splits['target_train'], splits['target_test'] = train_test_split(data, target, test_size=0.3, random_state=7) 
    return splits


def pickle_data(splits, p1, p2):
    for split in splits.keys():
        if p1 == "Stemmed" and p2 =="NoStopWords":
            pd.to_pickle(splits[split], os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, split+'Stemmed_NoStopWords'+'.pkl'))
            print()
            print('data frame for "' + split + '":')
            print()
            print(splits[split].head())
        elif p1 == "Lemmatized" and p2 =="NoStopWords":
            pd.to_pickle(splits[split], os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, split+'Lemmatized_NoStopWords'+'.pkl'))
            print()
            print('data frame for "' + split + '":')
            print()
            print(splits[split].head())
        elif p1 == "Stemmed" and p2 =="":
            pd.to_pickle(splits[split], os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, split+'Stemmed_WithStopWords'+'.pkl'))
            print()
            print('data frame for "' + split + '":')
            print()
            print(splits[split].head())
        elif p1 == "Lemmatized" and p2 =="":
            pd.to_pickle(splits[split], os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, split+'Lemmatized_WithStopWords'+'.pkl'))
            print()
            print('data frame for "' + split + '":')
            print()
            print(splits[split].head())          
    print()


def pickle_dictionary(dictionary, p1, p2):
    if p1 == "Stemmed" and p2 =="NoStopWords":
        pd.to_pickle(dictionary, os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, 'Stemmed_NoStopWords_dictionary.pkl'))
    elif p1 == "Lemmatized" and p2 =="NoStopWords":
        pd.to_pickle(dictionary, os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, 'Lemmatized_NoStopWords_dictionary.pkl'))
    elif p1 == "Stemmed" and p2 =="":
        pd.to_pickle(dictionary, os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, 'Stemmed_WithStopWords_dictionary.pkl'))
    elif p1 == "Lemmatized" and p2 =="":
        pd.to_pickle(dictionary, os.path.join(ARGS.path_data_preprocessed, ARGS.dataset, 'Lemmatized_WithStopWords_dictionary.pkl'))

def main(ARGS):
    '''Main body of the code'''
    # get dataset as a data frame
    
    df = get_data()
    
    print()
    print('example review before any preprocessing:')
    print()
    print(df['review'][3])
    # clean data
    df['Stemmed_and_NoStopWords'] = clean_data(df['review'],'Stemming', 'NoStopWords')
    df['Lemmatized_and_NoStopWords'] = clean_data(df['review'],'Lemmatized', 'NoStopWords')
    df['Stemmed_and_StopWords'] = clean_data(df['review'],'Stemming', '')
    df['Lemmatized_and_StopWords'] = clean_data(df['review'],'Lemmatized', '')
    print()
    print('example review after cleaning:')
    print()
    print(df['Stemmed_and_StopWords'][3])
    # encode data
    df['codes_for_Stemmed_and_NoStopWords'], dictionary1 = encode_data(df['Stemmed_and_NoStopWords'])
    df['codes_for_Lemmatized_and_NoStopWords'], dictionary2 = encode_data(df['Lemmatized_and_NoStopWords'])
    df['codes_for_Stemmed_and_StopWords'], dictionary3 = encode_data(df['Stemmed_and_StopWords'])
    df['codes_for_Lemmatized_and_StopWords'], dictionary4 = encode_data(df['Lemmatized_and_StopWords'])
    print()
    print('example review after encoding:')
    print()
    print(df['codes_for_Stemmed_and_StopWords'][3])
    print()
    print('dictionary size: ' + str(len(dictionary1)))
    print()
    dictionary_head = {k: dictionary1[k] for k in list(dictionary1)[:10]}
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
    df_splits1 = split_data(df, 'Stemmed', 'NoStopWords')
    df_splits2 = split_data(df, 'Lemmatized', 'NoStopWords')
    df_splits3 = split_data(df, 'Stemmed', '')
    df_splits4 = split_data(df, 'Lemmatized', '')
    # pickle data and dictionary
    pickle_data(df_splits1, 'Stemmed', 'NoStopWords')
    pickle_data(df_splits2, 'Lemmatized', 'NoStopWords')
    pickle_data(df_splits3, 'Stemmed', '')
    pickle_data(df_splits4, 'Lemmatized', '')
    
    pickle_dictionary(dictionary1, 'Stemmed', 'NoStopWords')
    pickle_dictionary(dictionary2, 'Lemmatized', 'NoStopWords')
    pickle_dictionary(dictionary3, 'Stemmed', '')
    pickle_dictionary(dictionary4, 'Lemmatized', '')    


def parse_arguments(parser):
    '''Read user arguments'''
    parser.add_argument('--path_data_raw', 
                        type=str,  
                        default='../data',  
                        help='Path to raw data')
    parser.add_argument('--path_data_preprocessed', 
                        type=str,  
                        default='data2',  
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