from retain_arguments import *


def read_data():
    def convert_input(data):
        return np.asarray(data)
    def convert_output(data):
        data = [[[target]] for target in data]
        return np.asarray(data)
    # read data from given paths
    x_train = convert_input(pd.read_pickle(ARGS.path['data_train'])['codes'].tolist())
    x_test = convert_input(pd.read_pickle(ARGS.path['data_test'])['codes'].tolist())
    y_train = convert_output(pd.read_pickle(ARGS.path['target_train'])['target'].tolist())
    y_test = convert_output(pd.read_pickle(ARGS.path['target_test'])['target'].tolist())
    # read dictionary from given path
    with open(ARGS.path['dictionary'], 'rb') as f:
        dictionary = pickle.load(f)
    return x_train, y_train, x_test, y_test, dictionary


def import_model(path):
    # prepare tensorflow session
    K.clear_session()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.compat.v1.Session(config=config)
    K.set_session(tfsess)
    # import model from given path
    model = load_model(path, custom_objects={'FreezePadding': FreezePadding})
    return model, model_with_attention


def partition_model(model):
    model_with_attention = Model(model.inputs, model.outputs +\
                                              [model.get_layer(name='dropout_1').output,\
                                               model.get_layer(name='alpha_softmax').output,\
                                               model.get_layer(name='beta_dense').output])


def get_model_parameters(model):
    # extract model arguments that were used during training
    class ModelParameters:
        # helper class to store model parameters in the same format as ARGS
        def __init__(self):
            self.num_codes = None
            self.emb_weights = None
            self.output_weights = None
            self.bias = None
    params = ModelParameters()
    params.num_codes = model.get_layer(name='embedding').input_dim-1
    params.emb_weights = model.get_layer(name='embedding').get_weights()[0]
    params.output_weights, params.bias = model.get_layer(name='time_distributed').get_weights()
    print('\nModel bias: {}'.format(params.bias))
    return params


def main(ARGS):
    print('>>> Reading dictionary and data as numpy arrays')
    x_train, y_train, x_test, y_test, dictionary = read_data()
    print('\nShape of "x_train": ' + str(x_train.shape))
    print('Shape of "y_train": ' + str(y_train.shape))
    print('Shape of "x_test": ' + str(x_test.shape))
    print('Shape of "y_test": ' + str(y_test.shape))

    print('\n>>> Loading original model')
    model, model_with_attention = import_model(ARGS.path['model'])

    print('\n>>> Extracting parameters of original')
    model_parameters = get_model_parameters(model)


if __name__ == '__main__':
    print('\n>>> Initialize arguments')
    ARGS = Arguments(dataset='IMDB', dir_data='data', dir_model='model', 
                    preprocessing='lemmatize', stopwords='remove',
                    num_codes=100000, num_sentences=50, num_words=50,
                    emb_size=200, alpha_rec_size=200, beta_rec_size=200, 
                    dropout_input=0.0, dropout_context=0.0, l2=0.0,
                    epochs=1, batch_size=128)
    main(ARGS)