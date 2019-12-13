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


def get_alphas(i, ra, rb, rx, model_parameters, dictionary):
    # importances = []
    # df = pd.DataFrame()
    for j in range(len(rx)):
        sent_codes = rx[j]
        sent_alpha = ra[j][0]
        sent_beta = rb[j]
        values = np.full(fill_value='Positive', shape=(len(sent_codes),))
        values_mask = np.array([1. if value == 'Positive' else value for value in values], dtype=np.float32)
        beta_scaled = sent_beta * model_parameters.emb_weights[sent_codes]
        output_scaled = np.dot(beta_scaled, model_parameters.output_weights)
        alpha_scaled = values_mask * sent_alpha * output_scaled
        
    # zero out highest attended sentence and hence all its words
    flat_output_scaled = [v for arr in ra for v in arr]
    max_index = np.where(flat_output_scaled == np.amax(flat_output_scaled))[0][0]
    ra_zero_high = copy.deepcopy(ra)
    ra_zero_high[max_index] = [0]
    ra_zero_high = [ra_zero_high / ra_zero_high.sum(axis=0)]

    # zero out random sentence and hence all its words
    ra_zero_rand = copy.deepcopy(ra)
    ra_zero_rand[randrange(0, len(ra), 1)] = [0]
    ra_zero_rand = [ra_zero_rand / ra_zero_rand.sum(axis=0)]

    # shuffle weights
    ra_perm = [np.random.shuffle(ra)]

    # uniformly distributed sentence attention
    ra_unif = np.full((1,ra.shape[0],1), 1 / ra.shape[0])

    # random uniform sentence attention
    random = np.random.uniform(size=(ra.shape[0],1))
    ra_rand = [random / random.sum(axis=0)]

    # save interpretation for review i, sentence j
    # if j == 3 and i == 0:
    #     print('\nNote attention weights for one review ' + str(i) + ', sentence ' + str(j))
    #     df = pd.DataFrame({'word': [dictionary[index] for index in sent_codes],
    #                              'importance_word': alpha_scaled[:, 0],
    #                              'importance_sentence': sent_alpha},
    #                               columns=['word', 'importance_word', 'importance_sentence'])
    #     df = df[df['word'] != '<PAD>']
    #     df.sort_values(['importance_word'], ascending=False, inplace=True)
    #     print(df)
    #     # importances.append(df)

    # save interpretation for review i, sentence j
    # if j == 3:
    #     print('\nNote attention weights for one review ' + str(i) + ', sentence ' + str(j))
    #     df = pd.DataFrame({'word': [dictionary[index] for index in sent_codes],
    #                               'importance_word': alpha_scaled[:, 0],
    #                               'importance_sentence': sent_alpha},
    #                               columns=['word', 'importance_word', 'importance_sentence'])
    #     df = df[df['word'] != '<PAD>']
    #     df.sort_values(['importance_word'], ascending=False, inplace=True)
    #     print(df)
    #     # importances.append(df)

    return ra_zero_high[0], ra_zero_rand[0], ra_perm[0], ra_rand[0], ra_unif[0]


def modify_weights(x_data, y_data, old_pred, old_drops, old_alphas, old_betas, model_parameters, dictionary):
    alphas_type = ['orig','zero_high','zero_rand','perm','rand','unif']
    alphas_dict = {a: [] for a in alphas_type}
    # for i in range(7):
    for i in range(len(x_data)):
        rx = x_data[i]
        ry = y_data[i]
        rp = old_pred[i]
        ra = old_alphas[i]
        rb = old_betas[i]
        ra_zero_high, ra_zero_rand, ra_perm, ra_rand, ra_unif = get_alphas(i, ra, rb, rx, model_parameters, dictionary)
        alphas_dict['orig'].append(ra)
        alphas_dict['zero_high'].append(ra_zero_high)
        alphas_dict['zero_rand'].append(ra_zero_rand)
        alphas_dict['perm'].append(ra_perm)
        alphas_dict['rand'].append(ra_rand)
        alphas_dict['unif'].append(ra_unif)
    # convert all to numpy arrays
    for a in alphas_dict.keys():
        alphas_dict[a] = np.array(alphas_dict[a])
    return alphas_dict


def main(ARGS):
    print('\n>>> Reading dictionary and data as numpy arrays')
    x_train, y_train, x_test, y_test, dictionary = read_data()
    y_test_binary = [pred[0][0] for pred in y_test]
    print('\nShape of "x_train": ' + str(x_train.shape))
    print('Shape of "y_train": ' + str(y_train.shape))
    print('Shape of "x_test": ' + str(x_test.shape))
    print('Shape of "y_test": ' + str(y_test.shape))

    print('\n>>> Loading original model')
    model = import_model(ARGS.path['model'])

    print('\n>>> Extracting parameters of original')
    model_parameters = get_model_parameters(model)

    print('\n>>> Partition model at the attention layer (softmax_1, beta_dense_0, dropout_1)')
    submodel_1, submodel_2 = partition_model(model)

    print('\n>>> Predicting probabilities with original model')
    preds = model.predict(x_test)
    preds = [pred[0][0] for pred in preds]
    preds_binary = [int(round(pred)) for pred in preds]
    print('\nPredictions (float):')
    print(preds)
    print('\nPredictions (int):')
    print(preds_binary)

    print('\n>>> Gather attention weights from original model to feed as input for second submodel')
    drops, alphas, betas = submodel_1.predict(x_test)
    print('\nShapes of original drops, alphas, betas respectively:')
    print(drops.shape, alphas.shape, betas.shape)

    print('\n>>> Modifying alpha attention weights') 
    alphas = modify_weights(x_test, y_test, preds, drops, alphas, betas, model_parameters, dictionary)
    print('\nShapes of original and modified alphas (orig, zero, perm, rand, unif):')
    print(alphas['orig'].shape, alphas['zero_high'].shape, alphas['zero_rand'].shape,\
          alphas['perm'].shape, alphas['rand'].shape, alphas['unif'].shape)

    # TODO: REMOVE TEMPORARY MODIFICATION when reseting to 15000 instead of 10; and implementing perm
    perm = alphas.pop("perm", None)
    y_test_binary = [pred[0][0] for pred in y_test]
    # y_test_binary = y_test_binary[:len(alphas['orig'])]
    # preds_binary = preds_binary[:len(alphas['orig'])]
    # preds = preds[:len(alphas['orig'])]
    # drops = drops[:len(alphas['orig'])]
    # betas = betas[:len(alphas['orig'])]
    # print('\nShapes of new drops, alphas, betas respectively:')
    # print(drops.shape, alphas['orig'].shape, betas.shape)
    # print('\nShapes of preds, preds_binary, y_test_binary respectively:')
    # print(len(preds), len(preds_binary), len(y_test_binary))

    print('\n>>> Predicting probabilities with second submodel and alpha weights')
    results = pd.DataFrame(index=alphas.keys(), columns=['preds','preds_binary','acc','JSdiv','JSdiv_binary','JSdiv_dist'])
    print('\nCurrently empty dataframe to store results:')
    print(results)

    for a in alphas.keys():
        print('\nAlpha weights: ' + a)
        
        # note predictions
        new_preds = submodel_2.predict([drops, alphas[a], betas])
        new_preds = [pred[0][0] for pred in new_preds]
        results.at[a,'preds'] = new_preds
        print('\nPredictions (float):')
        print(new_preds)

        # note predictions (binary)
        new_preds_binary = [int(round(pred)) for pred in new_preds]
        results.at[a,'preds_binary'] = new_preds_binary
        print('Predictions (int):')
        print(new_preds_binary)

        # note testing accuracy
        acc = accuracy_score(y_test_binary, new_preds_binary)
        results.at[a,'acc'] = acc
        print('Accuracy: ' + str(acc))

        # note JS divergence 
        JSdiv = JS(preds, new_preds)**2
        results.at[a,'JSdiv'] = JSdiv
        print('\nJS divergence: ' + str(JSdiv))

        # note JS divergence (binary)
        JSdiv_binary = JS(preds_binary, new_preds_binary)**2
        results.at[a,'JSdiv_binary'] = JSdiv_binary
        print('JS divergence (binary): ' + str(JSdiv_binary))

        # note JS divergence (distributions)
        dist = np.array([preds_binary.count(0), preds_binary.count(1)]) / (preds_binary.count(0) + preds_binary.count(1))
        new_dist = np.array([new_preds_binary.count(0), new_preds_binary.count(1)]) / (new_preds_binary.count(0) + new_preds_binary.count(1))
        JSdiv_dist = JS(dist, new_dist)**2
        results.at[a,'JSdiv_dist'] = JSdiv_dist
        print('JS divergence (distribution of 0s and 1s): ' + str(JSdiv_binary))

        # note decision flips
        differences = [preds_binary[i] - new_preds_binary[i] for i in range(len(new_preds_binary))]
        additions = [preds_binary[i] + new_preds_binary[i] for i in range(len(new_preds_binary))]
        print('\nDecisions flips')
        print(differences)
        print(additions)
        print('Old: 0, New: 0 (same) --- ' + str(additions.count(0)))
        print('Old: 0, New: 1 (flip) --- ' + str(differences.count(-1)))
        print('Old: 1, New: 1 (same) --- ' + str(additions.count(2)))
        print('Old: 1, New: 0 (flip) --- ' + str(differences.count(1)))
        print('Double checking all data points: ' + str(additions.count(0) + additions.count(2) + differences.count(-1) + differences.count(1)))

    # note decision flips specific to i* and r*
    print('\n>>> Noting decisions flips for i* and r*')
    i_preds = submodel_2.predict([drops, alphas['zero_high'], betas])
    i_preds = [pred[0][0] for pred in i_preds]
    i_preds_binary = [int(round(pred)) for pred in i_preds]
    i_flips = [abs(preds_binary[i] - i_preds_binary[i]) for i in range(len(i_preds_binary))]
    r_preds = submodel_2.predict([drops, alphas['zero_rand'], betas])
    r_preds = [pred[0][0] for pred in r_preds]
    r_preds_binary = [int(round(pred)) for pred in r_preds]
    r_flips = [abs(preds_binary[i] - r_preds_binary[i]) for i in range(len(r_preds_binary))]
    print(i_flips)
    print(r_flips)
    count_yy = 0
    count_yn = 0
    count_ny = 0
    count_nn = 0
    for i in range(len(i_flips)):
        if i_flips[i] == 1 and r_flips[i] == 1:
            count_yy += 1
        if i_flips[i] == 1 and r_flips[i] == 0:
            count_yn += 1
        if i_flips[i] == 0 and r_flips[i] == 1:
            count_ny += 1
        if i_flips[i] == 0 and r_flips[i] == 0:
            count_nn += 1
    print('i*: yes, r: yes --- ' + str(count_yy))
    print('i*: yes, r: non --- ' + str(count_yn))
    print('i*: non, r: yes --- ' + str(count_ny))
    print('i*: non, r: non --- ' + str(count_nn))

    print('\nFilled dataframe of results:')
    results.to_csv(index=True, path_or_buf='results.csv')
    print(results)


if __name__ == '__main__':
    print('\n>>> Initialize arguments')
    ARGS = Arguments(dataset='IMDB', dir_data='data', dir_model='model', 
                    preprocessing='lemmatize', stopwords='include',
                    num_codes=100000, num_sentences=50, num_words=50,
                    emb_size=200, alpha_rec_size=200, beta_rec_size=200, 
                    dropout_input=0.0, dropout_context=0.0, l2=0.0,
                    epochs=1, batch_size=128)
    main(ARGS)