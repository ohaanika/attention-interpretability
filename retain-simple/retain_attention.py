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
    return model


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


def partition_model(model):
    # define first submodel to retrieve weights from attention layers of original model
    submodel_1 = Model(model.inputs, [model.get_layer(name='dropout_1').output,\
                                    model.get_layer(name='alpha_softmax').output,\
                                    model.get_layer(name='beta_dense').output])
    # take dropout_1, alpha_softmax, beta_dense from original model as inputs for second submodel 
    new_inputs = []
    for layer_name in ['dropout_1', 'alpha_softmax', 'beta_dense']:
        new_inputs.append(L.Input(
            shape=model.get_layer(layer_name).output_shape[1:],
            name=layer_name))
    # stack sequential layers until output from original model
    output = new_inputs
    for layer_name in ['multiply', 'lambda_2', 'lambda_3', 'dropout_2', 'time_distributed']:
        layer = model.get_layer(layer_name)
        output = layer(output)
    new_outputs = [output]
    # define the second submodel with appropriate inputs and outputs
    submodel_2 = Model(inputs=new_inputs, outputs=new_outputs)
    # compile second submodel
    submodel_2.compile(
        optimizer='adamax',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        sample_weight_mode='temporal')
    # visually inspect architecture of second submodel
    print()
    print(submodel_1.summary())
    print()
    print(submodel_2.summary())
    return submodel_1, submodel_2


def main(ARGS):
    print('>>> Reading dictionary and data as numpy arrays')
    x_train, y_train, x_test, y_test, dictionary = read_data()
    print('\nShape of "x_train": ' + str(x_train.shape))
    print('Shape of "y_train": ' + str(y_train.shape))
    print('Shape of "x_test": ' + str(x_test.shape))
    print('Shape of "y_test": ' + str(y_test.shape))

    print('\n>>> Loading original model')
    model = import_model(ARGS.path['model'])

    print('\n>>> Extracting parameters of original')
    model_parameters = get_model_parameters(model)

    print('>>> Partition model at the attention layer (softmax_1, beta_dense_0, dropout_1)')
    submodel_1, submodel_2 = partition_model(model)

    print('>>> Predicting probabilities with original model')
    y_pred = model.predict(x_test)
    y_pred = [pred[0][0] for pred in y_pred]
    y_pred_binary = [int(round(pred)) for pred in y_pred]
    print('\nPredictions (float):')
    print(y_pred)
    print('\nPredictions (int):')
    print(y_pred_binary)

    print('\n>>> Gather attention weights from original model to feed as input for second submodel')
    drops, alphas, betas = submodel_1.predict(x_test)
    print('\nShapes of y_pred, drops, alphas, betas respectively:')
    print(drops.shape, alphas.shape, betas.shape)

    print('\n>>> Predicting probabilities with second submodel (without modifying attention weights)')
    y_pred = submodel_2.predict([drops, alphas, betas])
    y_pred = [pred[0][0] for pred in y_pred]
    y_pred_binary = [int(round(pred)) for pred in y_pred]
    print('\nPredictions (float):')
    print(y_pred)
    print('\nPredictions (int):')
    print(y_pred_binary)


if __name__ == '__main__':
    print('\n>>> Initialize arguments')
    ARGS = Arguments(dataset='IMDB', dir_data='data', dir_model='model', 
                    preprocessing='lemmatize', stopwords='remove',
                    num_codes=100000, num_sentences=50, num_words=50,
                    emb_size=200, alpha_rec_size=200, beta_rec_size=200, 
                    dropout_input=0.0, dropout_context=0.0, l2=0.0,
                    epochs=1, batch_size=128)
    main(ARGS)