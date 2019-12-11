import os 
import re
import math
import pickle as pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras import layers as L
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import Constraint
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import plot_model


class Arguments():
    def __init__(self, dataset, dir_data, dir_model, preprocessing, stopwords,
                 emb_size, alpha_rec_size, beta_rec_size, dropout_input, dropout_context, l2,
                 epochs, batch_size):
        # directories
        self.directory = {
            'data': os.path.join(dir_data, preprocessing + '_' + stopwords),
            'model': dir_model
        }
        # paths
        self.path = {
            'data_raw': os.path.join('..', dir_data, dataset, dataset+'.txt'),
            'data_train': os.path.join(dir_data, dataset, preprocessing + '_' + stopwords + '_stopwords', 'data_train.pkl'),
            'data_test': os.path.join(dir_data, dataset, preprocessing + '_' + stopwords + '_stopwords', 'data_test.pkl'),
            'target_train': os.path.join(dir_data, dataset, preprocessing + '_' + stopwords + '_stopwords', 'target_train.pkl'),
            'target_test': os.path.join(dir_data, dataset, preprocessing + '_' + stopwords + '_stopwords', 'target_test.pkl'),
            'dictionary': os.path.join(dir_data, dataset, preprocessing + '_' + stopwords + '_stopwords', 'dictionary.pkl'),
            'model': os.path.join(dir_model, 'weights.01.hdf5'),
            'model_plot': os.path.join(dir_model, 'model.png'),
        }
        # preprocessing methods
        self.preprocessing = preprocessing # CHOICES: ['lemmatize', 'stem']
        self.stopwords = stopwords # CHOICES: ['remove', 'include']
        # vocabulary ids
        self.pad_id = 0
        self.oov_id = 1
        # number of words in vocabulary i.e. self.max_features? self.num_codes?
        self.num_codes = 150850
        self.max_features = 150850
        # maximum number of sentences/words after which the data is truncated i.e. self.n_steps?
        self.num_sentences = 50
        self.num_words = 50
        # size of embedding layer # HYPERPARAMETER
        self.emb_size = emb_size 
        # size of recurrent layers # HYPERPARAMETER
        self.alpha_rec_size = alpha_rec_size 
        self.beta_rec_size = beta_rec_size 
        # dropout rate for embedding # HYPERPARAMETER
        self.dropout_input = dropout_input
        # dropout rate for context vector # HYPERPARAMETER
        self.dropout_context = dropout_context
        # L2 regularitzation value # HYPERPARAMETER
        self.l2 = l2
        # number of epochs
        self.epochs = epochs
        # batch size
        self.batch_size = batch_size


class FreezePadding(Constraint):
    '''Freezes the last weight to be near 0.'''
    def __call__(self, w):
        other_weights = K.cast(K.ones(K.shape(w))[:-1], K.floatx())
        last_weight = K.cast(K.equal(K.reshape(w[-1, :], (1, K.shape(w)[1])), 0.), K.floatx())
        appended = K.concatenate([other_weights, last_weight], axis=0)
        w *= appended
        return w


# initialize arguments
ARGS = Arguments(dataset='IMDB', dir_data='data', dir_model='model', preprocessing='lemmatize', stopwords='remove',
                 emb_size=200, alpha_rec_size=200, beta_rec_size=200, dropout_input=0.0, dropout_context=0.0, l2=0.0,
                 epochs=1, batch_size=128)


# set random seed
# tf.random.set_random_seed(7)
# np.random.seed(7)