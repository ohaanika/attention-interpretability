import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.layers as L
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing import sequence
from keras.utils.data_utils import Sequence
from keras.regularizers import l2
from keras.constraints import non_neg, Constraint
from keras_exp.multigpu import get_available_gpus, make_parallel
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


class SequenceBuilder(Sequence):
    '''Generate batches of data'''
    def __init__(self, data, target, batch_size, ARGS, target_out=True):
        # Receive all appropriate data
        self.codes = data
        self.num_codes = ARGS.num_codes
        self.target = target
        self.batch_size = batch_size
        self.target_out = target_out
        self.n_steps = ARGS.n_steps

    def __len__(self):
        '''Compute number of batches and add extra batch if the data doesn't exactly divide into batches'''
        if len(self.codes)%self.batch_size == 0:
            return len(self.codes) // self.batch_size
        return len(self.codes) // self.batch_size+1

    def __getitem__(self, idx):
        '''Get batch of specific index'''
        def pad_data(data, length_visits, length_codes, pad_value=0):
            '''Pad data to desired number of visits and codes inside each visit'''
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
        pad_length_visits = min(max(map(len, x_codes)), self.n_steps)
        pad_length_codes = max(map(lambda x: max(map(len, x)), x_codes))
        # Number of elements in a batch (useful in case of partial batches)
        length_batch = len(x_codes)
        # Pad data
        x_codes = pad_data(x_codes, pad_length_visits, pad_length_codes, self.num_codes)
        outputs = [x_codes]

        # Add target if necessary (training vs validation)
        if self.target_out:
            target = self.target[batch_slice].reshape(length_batch, 1, 1)
            # In our experiments sample weights provided worse results
            return (outputs, target)

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


def read_data(ARGS):
    '''Read the data from provided paths and assign it into lists'''
    data_train = pd.read_pickle(ARGS.path_data_train)['codes'].values
    data_test = pd.read_pickle(ARGS.path_data_test)['codes'].values
    y_train = pd.read_pickle(ARGS.path_target_train)['target'].values
    y_test = pd.read_pickle(ARGS.path_target_test)['target'].values
    return (data_train, y_train, data_test, y_test)


def create_model(ARGS):
    '''Create and compile model and assign it to provided devices'''
    def retain(ARGS):
        '''Create the model'''
        # Define the constant for model saving
        reshape_size = ARGS.emb_size
        if ARGS.allow_negative:
            embeddings_constraint = FreezePadding()
            beta_activation = 'tanh'
            output_constraint = None
        else:
            embeddings_constraint = FreezePadding_Non_Negative()
            beta_activation = 'sigmoid'
            output_constraint = non_neg()

        # Get available gpus, returns empty list if none
        glist = get_available_gpus()

        def reshape(data):
            '''Reshape the context vectors to 3D vector'''
            return K.reshape(x=data, shape=(K.shape(data)[0], 1, reshape_size))

        # Code input
        codes = L.Input((None, None), name='codes_input')

        # Calculate embedding for each code and sum them to a visit level
        codes_embs_total = L.Embedding(ARGS.num_codes+1,
                                       ARGS.emb_size,
                                       name='embedding',
                                       embeddings_constraint=embeddings_constraint)(codes)
        codes_embs = L.Lambda(lambda x: K.sum(x, axis=2))(codes_embs_total)

        # Apply dropout on inputs
        codes_embs = L.Dropout(ARGS.dropout_input)(codes_embs)

        # If training on GPU and Tensorflow use CuDNNLSTM for much faster training
        if glist:
            alpha = L.Bidirectional(L.CuDNNLSTM(ARGS.recurrent_size, return_sequences=True), name='alpha')
            beta = L.Bidirectional(L.CuDNNLSTM(ARGS.recurrent_size, return_sequences=True), name='beta')
        else:
            alpha = L.Bidirectional(L.LSTM(ARGS.recurrent_size, return_sequences=True, implementation=2), name='alpha')
            beta = L.Bidirectional(L.LSTM(ARGS.recurrent_size, return_sequences=True, implementation=2), name='beta')

        alpha_dense = L.Dense(1, kernel_regularizer=l2(ARGS.l2))
        beta_dense = L.Dense(ARGS.emb_size, activation=beta_activation, kernel_regularizer=l2(ARGS.l2))

        # Compute alpha, visit attention
        alpha_out = alpha(codes_embs)
        alpha_out = L.TimeDistributed(alpha_dense, name='alpha_dense_0')(alpha_out)
        alpha_out = L.Softmax(axis=1)(alpha_out)
        # Compute beta, codes attention
        beta_out = beta(codes_embs)
        beta_out = L.TimeDistributed(beta_dense, name='beta_dense_0')(beta_out)
        # Compute context vector based on attentions and embeddings
        c_t = L.Multiply()([alpha_out, beta_out, codes_embs])
        c_t = L.Lambda(lambda x: K.sum(x, axis=1))(c_t)
        # Reshape to 3d vector for consistency between Many to Many and Many to One implementations
        contexts = L.Lambda(reshape)(c_t)

        # Make a prediction
        contexts = L.Dropout(ARGS.dropout_context)(contexts)
        output_layer = L.Dense(1, activation='sigmoid', name='dOut', kernel_regularizer=l2(ARGS.l2), 
                               kernel_constraint=output_constraint)

        # TimeDistributed is used for consistency between Many to Many and Many to One implementations
        output = L.TimeDistributed(output_layer, name='time_distributed_out')(contexts)
        # Define the model with appropriate inputs
        model = Model(inputs=[codes], outputs=[output])

        return model

    # Set Tensorflow to grow GPU memory consumption instead of grabbing all of it at once
    K.clear_session()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.compat.v1.Session(config=config)
    K.set_session(tfsess)
    # If there are multiple GPUs set up a multi-gpu model
    glist = get_available_gpus()
    if len(glist) > 1:
        with tf.device('/cpu:0'):
            model = retain(ARGS)
        model_final = make_parallel(model, glist)
    else:
        model_final = retain(ARGS)

    # Compile the model - adamax has produced best results in our experiments
    model_final.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'],
                        sample_weight_mode='temporal')

    return model_final


def create_callbacks(model, data, ARGS):
    '''Create the checkpoint and logging callbacks'''
    class LogEval(Callback):
        '''Logging callback'''
        def __init__(self, filepath, model, data, ARGS, interval=1):

            super(Callback, self).__init__()
            self.filepath = filepath
            self.interval = interval
            self.data_test, self.y_test = data
            self.generator = SequenceBuilder(data=self.data_test, target=self.y_test,
                                             batch_size=ARGS.batch_size, ARGS=ARGS,
                                             target_out=False)
            self.model = model
        def on_epoch_end(self, epoch, logs={}):
            # Compute ROC-AUC and average precision the validation data every interval epochs
            if epoch % self.interval == 0:
                # Compute predictions of the model
                y_pred = [x[-1] for x in
                          self.model.predict_generator(self.generator,
                                                       verbose=0,
                                                       use_multiprocessing=True,
                                                       workers=5,
                                                       max_queue_size=5)]
                score_roc = roc_auc_score(self.y_test, y_pred)
                score_pr = average_precision_score(self.y_test, y_pred)
                # Create log files if it doesn't exist, otherwise write to it
                if os.path.exists(self.filepath):
                    append_write = 'a'
                else:
                    append_write = 'w'
                with open(self.filepath, append_write) as file_output:
                    file_output.write('\nEpoch: {:d}- ROC-AUC: {:.6f} ; PR-AUC: {:.6f}'\
                            .format(epoch, score_roc, score_pr))

                print('\nEpoch: {:d} - ROC-AUC: {:.6f} PR-AUC: {:.6f}'\
                      .format(epoch, score_roc, score_pr))


    # Create callbacks
    checkpoint = ModelCheckpoint(filepath=ARGS.directory+'/weights.{epoch:02d}.hdf5')
    log = LogEval(ARGS.directory+'/log.txt', model, data, ARGS)
    return(checkpoint, log)


def train_model(model, data_train, y_train, data_test, y_test, ARGS):
    '''Train the model with appropriate callbacks and generator'''
    checkpoint, log = create_callbacks(model, (data_test, y_test), ARGS)
    train_generator = SequenceBuilder(data=data_train, target=y_train,
                                      batch_size=ARGS.batch_size, ARGS=ARGS)
    model.fit_generator(generator=train_generator, epochs=ARGS.epochs,
                        max_queue_size=15, use_multiprocessing=True,
                        callbacks=[checkpoint, log], verbose=1, workers=3, initial_epoch=0)


def main(ARGS):
    '''Main body of the code'''
    print('\n>>> Reading data')
    data_train, y_train, data_test, y_test = read_data(ARGS)
    print('\n>>> Creating model')
    model = create_model(ARGS)
    print('\n>>> Noting model summary')
    print(model.summary())
    print('\n>>> Training model')
    train_model(model=model, data_train=data_train, y_train=y_train, data_test=data_test, y_test=y_test, ARGS=ARGS)


def parse_arguments(parser):
    '''Read user arguments'''
    parser.add_argument('--num_codes',
                        type=int, 
                        default=150850, 
                        help='Number of medical codes')
    parser.add_argument('--emb_size', 
                        type=int, 
                        default=200,
                        help='Size of the embedding layer')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=1,
                        help='Number of epochs')
    parser.add_argument('--n_steps', 
                        type=int,  
                        default=300,
                        help='Maximum number of visits after which the data is truncated')
    parser.add_argument('--recurrent_size', 
                        type=int,  
                        default=200,
                        help='Size of the recurrent layers')
    parser.add_argument('--path_data_train',  
                        type=str,  
                        default='data/IMDB/data_train.pkl',
                        help='Path to train data')
    parser.add_argument('--path_data_test',  
                        type=str, 
                        default='data/IMDB/data_test.pkl',
                        help='Path to test data')
    parser.add_argument('--path_target_train',  
                        type=str, 
                        default='data/IMDB/target_train.pkl',
                        help='Path to train target')
    parser.add_argument('--path_target_test',  
                        type=str,  
                        default='data/IMDB/target_test.pkl',
                        help='Path to test target')
    parser.add_argument('--batch_size',  
                        type=int,  
                        default=32,
                        help='Batch Size')
    parser.add_argument('--dropout_input',  
                        type=float,  
                        default=0.0,
                        help='Dropout rate for embedding')
    parser.add_argument('--dropout_context',  
                        type=float,  
                        default=0.0,
                        help='Dropout rate for context vector')
    parser.add_argument('--l2', 
                        type=float, 
                        default=0.0,
                        help='L2 regularitzation value')
    parser.add_argument('--directory',  
                        type=str,  
                        default='model',
                        help='Directory to save the model and the log file to')
    parser.add_argument('--allow_negative',  
                        action='store_true',
                        help='If argument is present the negative weights for embeddings/attentions\
                         will be allowed (original RETAIN implementaiton)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)