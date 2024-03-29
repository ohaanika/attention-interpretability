'''RETAIN Model Evaluation'''
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score,\
                            precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.preprocessing import sequence
from keras.constraints import Constraint
from keras.utils.data_utils import Sequence
from keras_exp.multigpu import get_available_gpus, make_parallel


def import_model(path):
    '''Import model from given path and assign it to appropriate devices'''
    K.clear_session()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.Session(config=config)
    K.set_session(tfsess)
    model = load_model(path, custom_objects={'FreezePadding':FreezePadding,
                                             'FreezePadding_Non_Negative':FreezePadding_Non_Negative})
    if len(get_available_gpus()) > 1:
        model = make_parallel(model)
    return model


def get_model_parameters(model):
    '''Extract model arguments that were used during training'''
    class ModelParameters:
        '''Helper class to store model parametesrs in the same format as ARGS'''
        def __init__(self):
            self.num_codes = None
            self.numeric_size = None
            self.use_time = None

    params = ModelParameters()
    names = [layer.name for layer in model.layers]
    params.num_codes = model.get_layer(name='embedding').input_dim-1
    if 'numeric_input' in names:
        params.numeric_size = model.get_layer(name='numeric_input').input_shape[2]
    else:
        params.numeric_size = 0
    if 'time_input' in names:
        params.use_time = True
    else:
        params.use_time = False
    return params


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


def precision_recall(y_true, y_prob, graph):
    '''Print Precision Recall Statistics and Graph'''
    average_precision = average_precision_score(y_true, y_prob)
    if graph:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.style.use('ggplot')
        plt.clf()
        plt.plot(recall, precision,
                 label='Precision-Recall Curve  (Area = %0.3f)' % average_precision)
        plt.xlabel('Recall: P(predicted+|true+)')
        plt.ylabel('Precision: P(true+|predicted+)')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc='lower left')
        print('Precision-Recall Curve saved to pr.png')
        plt.savefig('pr.png')
    else:
        print('Average Precision: %0.3f' % average_precision)


def probability_calibration(y_true, y_prob,graph):
    if graph:
        fig_index = 1
        name = 'My pred'
        n_bins = 20
        fig = plt.figure(fig_index, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_prob, n_bins=n_bins, normalize=True)

        ax1.plot(mean_predicted_value, fraction_of_positives,
                 label=name)

        ax2.hist(y_prob, range=(0, 1), bins=n_bins, label=name,
                 histtype='step', lw=2)

        ax1.set_ylabel('Fraction of Positives')
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc='lower right')
        ax1.set_title('Calibration Plots  (Reliability Curve)')

        ax2.set_xlabel('Mean predicted value')
        ax2.set_ylabel('Count')
        ax2.legend(loc='upper center', ncol=2)
        print('Probability Calibration Curves saved to calibration.png')
        plt.tight_layout()
        plt.savefig('calibration.png')


def lift(y_true, y_prob, graph):
    '''Print Precision Recall Statistics and Graph'''
    prevalence = sum(y_true)/len(y_true)
    average_lift = average_precision_score(y_true, y_prob) / prevalence
    if graph:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        lift_values = precision/prevalence
        plt.style.use('ggplot')
        plt.clf()
        plt.plot(recall, lift_values,
                 label='Lift-Recall Curve  (Area = %0.3f)' % average_lift)
        plt.xlabel('Recall: P(predicted+|true+)')
        plt.ylabel('Lift')
        plt.xlim([0.0, 1.0])
        plt.legend(loc='lower left')
        print('Lift-Recall Curve saved to lift.png')
        plt.savefig('lift')
    else:
        print('Average Lift: %0.3f' % average_lift)


def roc(y_true, y_prob, graph):
    '''Print ROC Statistics and Graph'''
    roc_auc = roc_auc_score(y_true, y_prob)
    if graph:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (Area = %0.3f)'% roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specifity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        print('ROC Curve saved to roc.png')
        plt.savefig('roc.png')
    else:
        print('ROC-AUC: %0.3f' % roc_auc)


def accuracy(y_true, y_prob, graph):
    '''Print Accuracy Statistics and Graph'''
    y_prob = y_prob.round()
    acc = accuracy_score(y_true, y_prob)
    if graph:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='Accuracy curve (Area = %0.3f)'% acc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        #plt.xlabel('x-axis (edit)')
        #plt.ylabel('y-axis (edit)')
        #plt.title('title (edit)')
        plt.legend(loc='lower right')
        print('Accuracy Curve saved to accuracy.png')
        plt.savefig('accuracy.png')
    else:
        print('Accuracy: %0.3f' % acc)


class SequenceBuilder(Sequence):
    '''Generate Batches of data'''
    def __init__(self, data, model_parameters, ARGS):
        #Receive all appropriate data
        self.codes = data[0]
        index = 1
        if model_parameters.numeric_size:
            self.numeric = data[index]
            index += 1

        if model_parameters.use_time:
            self.time = data[index]

        self.num_codes = model_parameters.num_codes
        self.batch_size = ARGS.batch_size
        self.numeric_size = model_parameters.numeric_size
        self.use_time = model_parameters.use_time
        self.n_steps = ARGS.n_steps

    def __len__(self):
        '''Compute number of batches.
        Add extra batch if the data doesn't exactly divide into batches
        '''
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
                        #Populate the data into the appropriate visit
                        mhot[:len(step)] = step

            return zeros
        #Compute reusable batch slice
        batch_slice = slice(idx*self.batch_size, (idx+1)*self.batch_size)
        x_codes = self.codes[batch_slice]
        #Max number of visits and codes inside the visit for this batch
        pad_length_visits = min(max(map(len, x_codes)), self.n_steps)
        pad_length_codes = max(map(lambda x: max(map(len, x)), x_codes))
        #Number of elements in a batch (useful in case of partial batches)
        length_batch = len(x_codes)
        #Pad data
        x_codes = pad_data(x_codes, pad_length_visits, pad_length_codes, self.num_codes)
        outputs = [x_codes]
        #Add numeric data if necessary
        if self.numeric_size:
            x_numeric = self.numeric[batch_slice]
            x_numeric = pad_data(x_numeric, pad_length_visits, self.numeric_size, -99.0)
            outputs.append(x_numeric)
        #Add time data if necessary
        if self.use_time:
            x_time = sequence.pad_sequences(self.time[batch_slice],
                                            dtype=np.float32, maxlen=pad_length_visits,
                                            value=+99).reshape(length_batch, pad_length_visits, 1)
            outputs.append(x_time)

        return outputs


def read_data(model_parameters, ARGS):
    '''Read the data from provided paths and assign it into lists'''
    data = pd.read_pickle(ARGS.path_data)
    y = pd.read_pickle(ARGS.path_target)['target'].values
    data_output = [data['codes'].values]

    if model_parameters.numeric_size:
        data_output.append(data['numerics'].values)
    if model_parameters.use_time:
        data_output.append(data['to_event'].values)
    return (data_output, y)


def get_predictions(model, data, model_parameters, ARGS):
    '''Get Model Predictions'''
    test_generator = SequenceBuilder(data, model_parameters, ARGS)
    preds = model.predict_generator(generator=test_generator, max_queue_size=15,
                                    use_multiprocessing=True, verbose=1, workers=3)
    return preds


def main(ARGS):
    '''Main body of the code'''
    print('Loading Model and Extracting Parameters')
    model = import_model(ARGS.path_model)
    model_parameters = get_model_parameters(model)
    print('Reading Data')
    data, y = read_data(model_parameters, ARGS)
    print('Predicting the probabilities')
    probabilities = get_predictions(model, data, model_parameters, ARGS)
    print('Evaluating')
    # TODO: test out printing accuracy, uncomment out other metrics later
    accuracy(y, probabilities[:, 0, -1], ARGS.save_graphs)
    roc(y, probabilities[:, 0, -1], ARGS.save_graphs)
    precision_recall(y, probabilities[:, 0, -1], ARGS.save_graphs)
    lift(y, probabilities[:, 0, -1], ARGS.save_graphs)
    probability_calibration(y, probabilities[:, 0, -1], ARGS.save_graphs)


def parse_arguments(parser):
    '''Read user arguments'''
    parser.add_argument('--path_model',
                        type=str,  
                        default='model/weights.01.hdf5',
                        help='Path to the model to evaluate')
    parser.add_argument('--path_data',  
                        type=str,  
                        default='data2/IMDB/data_testLemmatized_NoStopWords.pkl',
                        help='Path to evaluation data')
    parser.add_argument('--path_target',  
                        type=str,  
                        default='data2/IMDB/target_testLemmatized_NoStopWords.pkl',
                        help='Path to evaluation target')
    parser.add_argument('--save_graphs',  
                        action='store_true',
                        help='Output graphs if argument is present')
    parser.add_argument('--n_steps',  
                        type=int,  
                        default=300,
                        help='Maximum number of visits after which the data is truncated')
    parser.add_argument('--batch_size',  
                        type=int,  
                        default=32,
                        help='Batch size for prediction (higher values are generally faster)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)
