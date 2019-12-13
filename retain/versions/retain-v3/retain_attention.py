import os
import argparse
import pickle as pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.layers as L
import keras.backend as K
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing import sequence
from keras.constraints import non_neg, Constraint
from keras.utils.data_utils import Sequence
from keras.regularizers import l2
from keras_exp.multigpu import get_available_gpus, make_parallel
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


class SequenceBuilder(Sequence):
    '''Generate Batches of data'''
    def __init__(self, data, model_parameters, ARGS):
        # Receive all appropriate data
        self.codes = data
        self.num_codes = model_parameters.num_codes
        self.batch_size = ARGS.batch_size

    def __len__(self):
        '''Compute number of batches and add extra batch if the data doesn't exactly divide into batches'''
        if len(self.codes)%self.batch_size == 0:
            return len(self.codes) // self.batch_size
        return len(self.codes) // self.batch_size+1

    def __getitem__(self, idx):
        '''Get batch of specific index'''
        def pad_data(data, length_visits, length_codes, pad_value=0):
            '''Pad data to desired number of visiits and codes inside each visit'''
            zeros = np.full((len(data), length_visits, length_codes), pad_value)
            for steps, mat in zip(data, zeros):
                if steps != [[-1]]:
                    for step, mhot in zip(steps, mat[-len(steps):]):
                        # Populate the data into the appropriate visit
                        mhot[:len(step)] = step

            return zeros
        # Compute reusable batch slice
        batch_slice = slice(idx*self.batch_size, (idx+1)*self.batch_size)
        x_codes = self.codes[batch_slice]
        # Max number of visits and codes inside the visit for this batch
        pad_length_visits = max(map(len, x_codes))
        pad_length_codes = max(map(lambda x: max(map(len, x)), x_codes))
        # Pad data
        x_codes = pad_data(x_codes, pad_length_visits, pad_length_codes, self.num_codes)
        outputs = [x_codes]
        return outputs


class FreezePadding_Non_Negative(Constraint):
    '''Freezes the last weight to be near 0 and prevents non-negative embeddings'''
    def __call__(self, w):
        other_weights = K.cast(K.greater_equal(w, 0)[:-1], K.floatx())
        last_weight = K.cast(K.equal(K.reshape(w[-1, :], (1, K.shape(w)[1])), 0.), K.floatx())
        appended = K.concatenate([other_weights, last_weight], axis=0)
        w *= appended
        return w


class FreezePadding(Constraint):
    '''Freezes the last weight to be near 0.'''
    def __call__(self, w):
        other_weights = K.cast(K.ones(K.shape(w))[:-1], K.floatx())
        last_weight = K.cast(K.equal(K.reshape(w[-1, :], (1, K.shape(w)[1])), 0.), K.floatx())
        appended = K.concatenate([other_weights, last_weight], axis=0)
        w *= appended
        return w


def import_model(path):
    '''Import model from given path and assign it to appropriate devices'''
    K.clear_session()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.compat.v1.Session(config=config)
    K.set_session(tfsess)
    model = load_model(path, custom_objects={'FreezePadding':FreezePadding,
                                             'FreezePadding_Non_Negative':FreezePadding_Non_Negative})
    model_with_attention = Model(model.inputs, model.outputs +\
                                              [model.get_layer(name='softmax_1').output,\
                                               model.get_layer(name='beta_dense_0').output])
    return model, model_with_attention


model, model_with_attention = import_model(ARGS.path_model)