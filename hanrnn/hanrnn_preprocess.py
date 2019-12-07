import argparse
import os 
import re
import json
from random import random
from random import shuffle
import numpy as np
import pandas as pd
import nltk
import collections
from sklearn.model_selection import train_test_split
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords')


def get_data():
    data = pd.read_csv(os.path.join(DATA_PATH, DATA_SET, DATA_SET+'.txt'), sep='\t', header=None)
    data.columns = ['review', 'target']
    return data


def clean_data(data):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    clean_data = []
    for review in data:
        # remove html tags
        review = re.sub(re.compile('<.*?>'), '\n', review)
        # convert to lowercase
        review = review.lower()
        # keep only alphanumeric and space characters
        review = re.sub(r'[^\w\s]', '', review)
        # remove numeric characters
        review = re.sub(r'[0-9]+', '', review)
        # remove spaces at start and end of reviews
        review = review.strip()
        # replace all whitespaces and newlines with one space character
        review = re.sub(r'\s+', ' ', review)
        # tokenize review into words
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(review)
        # remove stop words
        filtered_words = [w for w in tokens if len(w) > 2 if not w in stop_words]
        # lemmatize words
        lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
        # concatenate words into review
        clean_review = ' '.join(lemmatized_words)
        # append clean review
        clean_data.append(clean_review)
    return clean_data


def write_list_of_instances_to_file(fname, instances, first_line):
    with open(fname, 'w') as f:
        if not first_line.endswith('\n'):
            first_line = first_line + '\n'
        f.write(first_line)
        for instance in instances:
            if not instance.endswith('\n'):
                instance = instance + '\n'
            f.write(instance)
    print("Done writing instances to " + fname)
    

def make_train_dev_test_sets(full_data_filename, pct_train_instances, pct_dev_instances,
                             training_fname, dev_fname, test_fname):
    with open(full_data_filename, 'r') as f:
        num_instances = -1
        for line in f:
            if line.strip() == '':
                continue
            num_instances += 1
        print("Found " + str(num_instances) + " instances in total.")

    training_inds = []
    dev_inds = []
    test_inds = []

    for i in range(num_instances):
        decider = random()
        if decider < pct_train_instances:
            training_inds.append(i)
        elif decider < pct_train_instances + pct_dev_instances:
            dev_inds.append(i)
        else:
            test_inds.append(i)

    training_instances = []
    dev_instances = []
    test_instances = []
    first_line = None
    instance_counter = 0
    with open(full_data_filename, 'r') as f:
        for line in f:
            if first_line is None:
                first_line = line
            else:
                if line.strip() == '':
                    continue
                if len(training_inds) > 0 and instance_counter == training_inds[0]:
                    training_instances.append(line)
                    del training_inds[0]
                elif len(dev_inds) > 0 and instance_counter == dev_inds[0]:
                    dev_instances.append(line)
                    del dev_inds[0]
                elif len(test_inds) > 0 and instance_counter == test_inds[0]:
                    test_instances.append(line)
                    del test_inds[0]
                instance_counter += 1

    assert len(training_inds) == 0
    assert len(dev_inds) == 0
    assert len(test_inds) == 0

    shuffle(training_instances)
    shuffle(dev_instances)
    shuffle(test_instances)

    print("Collected " + str(len(training_instances)) + " training instances.")
    print("Collected " + str(len(dev_instances)) + " dev instances.")
    print("Collected " + str(len(test_instances)) + " test instances.")

    write_list_of_instances_to_file(training_fname, training_instances, first_line)
    write_list_of_instances_to_file(dev_fname, dev_instances, first_line)
    write_list_of_instances_to_file(test_fname, test_instances, first_line)


if __name__ == '__main__':

    # define parser to parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='data', help='define path to data')
    parser.add_argument('--dataset', type=str, choices=['IMDB', 'yelp'], default='IMDB', help='select dataset')
    args = parser.parse_args()

    # define constants
    DATA_PATH = args.datapath
    DATA_SET = args.dataset
    DATA_SPLITS = ['train', 'test']

    # filenames
    imdb_data_file = os.path.join('data', 'IMDB', 'IMDB.txt')
    imdb_output_full_data_filename = os.path.join('data', 'IMDB', 'IMDB.tsv')
    imdb_output_train_filename = os.path.join('data', 'IMDB', 'IMDB_train.tsv')
    imdb_output_dev_filename = os.path.join('data', 'IMDB', 'IMDB_dev.tsv')
    imdb_output_test_filename = os.path.join('data', 'IMDB', 'IMDB_test.tsv')

    # get dataset as a data frame
    df = get_data()

    # clean data
    df['review'] = clean_data(df['review'])

    # TODO: add ids temporarily and reorder, but remove later?
    # df['id'] = range(df.shape[0])
    # df = df[['id', 'review', 'target']]

    # TODO: check dataframe, but remove later?
    print(df.head())

    # save as tsv file
    df.to_csv(imdb_output_full_data_filename, sep = '\t')

    # split data
    make_train_dev_test_sets(imdb_output_full_data_filename, .6, .1, imdb_output_train_filename,
                             imdb_output_dev_filename, imdb_output_test_filename)

