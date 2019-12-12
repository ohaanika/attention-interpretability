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
    return x_train, y_train, x_test, y_test


def define_model():
    # note contants
    embeddings_constraint = FreezePadding()
    beta_activation = 'tanh'
    output_constraint = None
    # define inputs and sequential embedding/lambda/dropout layers
    codes = L.Input(
        shape=(None, None), 
        name='codes')
    codes_embs_total = L.Embedding(
        input_dim=ARGS.num_codes+1,
        output_dim=ARGS.emb_size,
        embeddings_constraint=embeddings_constraint,
        name='embedding')(codes)
    codes_embs = L.Lambda(
        function=lambda x: K.sum(x, axis=2),
        name='lambda_1')(codes_embs_total)
    codes_embs = L.Dropout(
        rate=ARGS.dropout_input,
        name='dropout_1')(codes_embs)
    # define alpha layer to compute sentence level attention
    alpha = L.Bidirectional(
        layer=L.CuDNNLSTM(
            ARGS.alpha_rec_size, 
            return_sequences=True), 
        name='alpha')(codes_embs)
    alpha_dense = L.TimeDistributed(
        layer=L.Dense(
            units=1, 
            kernel_regularizer=l2(ARGS.l2)), 
        name='alpha_dense')(alpha)
    alpha_softmax = L.Softmax(
        axis=1,
        name='alpha_softmax')(alpha_dense)
    # define beta layer to compute word level attention
    beta = L.Bidirectional(
        layer=L.CuDNNLSTM(
            ARGS.beta_rec_size, 
            return_sequences=True), 
        name='beta')(codes_embs)
    beta_dense = L.TimeDistributed(
        L.Dense(
            units=ARGS.emb_size, 
            activation=beta_activation, 
            kernel_regularizer=l2(ARGS.l2)), 
        name='beta_dense')(beta)
    # define context layers to compute context vector based on attentions and embeddings
    c_t = L.Multiply(
        name='multiply')([codes_embs, alpha_softmax, beta_dense])
    c_t = L.Lambda(
        function=lambda x: K.sum(x, axis=1),
        name='lambda_2')(c_t)
    # reshape context vectors to 3d vector for consistency between Many to Many and Many to One implementations
    contexts = L.Lambda(
        function=lambda x: K.reshape(x, shape=(K.shape(x)[0], 1, 200)), # TODO: ARGS.emb_size = 200
        name='lambda_3')(c_t)
    # define dropout layer to make a prediction
    contexts = L.Dropout(
        rate=ARGS.dropout_context,
        name='dropout_2')(contexts)
    # TimeDistributed is used for consistency between Many to Many and Many to One implementations
    output = L.TimeDistributed(
        layer=L.Dense(
            units=1, 
            activation='sigmoid', 
            name='dOut', 
            kernel_regularizer=l2(ARGS.l2),
            kernel_constraint=output_constraint), 
        name='time_distributed')(contexts)
    # define the model with appropriate inputs and outputs
    model = Model(inputs=[codes], outputs=[output])
    # compile model 
    model.compile(
        optimizer='adamax',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        sample_weight_mode='temporal')
    # visually inspect architecture of model
    model.summary()
    return model


def main(ARGS):
    print('>>> Reading data as numpy arrays')
    x_train, y_train, x_test, y_test = read_data()
    print('\nShape of "x_train": ' + str(x_train.shape))
    print('Shape of "y_train": ' + str(y_train.shape))
    print('Shape of "x_test": ' + str(x_test.shape))
    print('Shape of "y_test": ' + str(y_test.shape))

    # TODO: pick best optimizer/loss/etc when compiling
    # https://keras.io/models/model/#compile
    # https://keras.io/optimizers/
    # https://keras.io/losses/
    # adamax has produced best results in RETAIN experiments
    print('>>> Create a model to take codes as input, targets as output')
    model = define_model()

    print('>>> Saving model architecture as ' + ARGS.path['model_plot'])
    plot_model(model, to_file=ARGS.path['model_plot'])

    # TODO: implement validation split to pick epochs, batch_size, and more
    # https://keras.io/models/model/#fit
    print('>>> Fitting model')
    checkpoint = ModelCheckpoint(filepath=os.path.join(ARGS.directory['model'], 'weights.{epoch:02d}.hdf5'))
    model.fit(x_train, y_train, epochs=ARGS.epochs, batch_size=ARGS.batch_size, callbacks=[checkpoint])

    # # compute loss and accuracy for train set
    result = model.evaluate(x_train, y_train)
    print(result)

    # # compute loss and accuracy for test set
    result = model.evaluate(x_test, y_test)
    print(result)


if __name__ == '__main__':
    print('\n>>> Initialize arguments')
    ARGS = Arguments(dataset='IMDB', dir_data='data', dir_model='model', 
                    preprocessing='lemmatize', stopwords='remove',
                    num_codes=100000, num_sentences=50, num_words=50,
                    emb_size=200, alpha_rec_size=200, beta_rec_size=200, 
                    dropout_input=0.0, dropout_context=0.0, l2=0.0,
                    epochs=1, batch_size=128)
    main(ARGS)