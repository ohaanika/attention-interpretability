from retain_arguments import *


def read_data():
    # read raw data from .txt file
    data = pd.read_csv(ARGS.path['data_raw'], sep='\t', header=None)
    # name columns
    data.columns = ['review', 'target']
    return data


def clean_data(data):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer =  PorterStemmer()
    print('\nPreprocessing: ' + ARGS.preprocessing + ', ' + ARGS.stopwords + ' stopwords')
    # parse through reviews in data
    clean_data = []
    for review in data:
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
            # test out different preprocessing methods
            if ARGS.preprocessing == 'lemmatize':   
                if ARGS.stopwords == 'remove':
                    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
                else:
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
            elif ARGS.preprocessing == 'remove':
                if ARGS.stopwords:
                    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
                else:
                    tokens = [stemmer.stem(t) for t in tokens]
            # concatenate words into sentence
            clean_sentence = ' '.join(tokens)
            # append clean sentence if its list is not empty
            if len(clean_sentence):
                clean_review.append(clean_sentence)
        # append clean review
        clean_data.append(clean_review)
    return clean_data


def split_data(df):
    splits = {}
    target = pd.DataFrame(data = np.asarray(df['target']), columns=['target'])
    data = pd.DataFrame(data = np.asarray(df['codes']), columns=['codes'])
    splits['data_train'], splits['data_test'], splits['target_train'], splits['target_test'] = train_test_split(data, target, test_size=0.3, random_state=7)
    return splits


def encode_data(splits):
    # initialize tokenizer with vocabulary length specified
    tokenizer = Tokenizer(num_words=ARGS.num_codes, oov_token='<OOV>')
    # initialize data
    data_train = splits['data_train']['codes']
    data_test = splits['data_test']['codes']
    # create internal vocabulary based on train data
    flat_data_train = [sentence for review in data_train for sentence in review]
    tokenizer.fit_on_texts(flat_data_train)
    # transform each sentence in review to a sequence of integers
    encoded_data_train = data_train.map(lambda s: tokenizer.texts_to_sequences(s))
    encoded_data_test = data_test.map(lambda s: tokenizer.texts_to_sequences(s))
    # convert dictionary to desired format
    word2id = tokenizer.word_index
    word2id['<PAD>'] = ARGS.pad_id
    word2id = sorted(word2id.items(), key=lambda x: x[1])
    id2word = {id: word for word, id in word2id}
    # replace data with encoded data
    splits['data_train']['codes'] = encoded_data_train
    splits['data_test']['codes'] = encoded_data_test
    return splits, id2word


def pad_data(splits):
    # initialize data (train and test combined) as a list
    data_both = splits['data_train']['codes'].tolist() + splits['data_test']['codes'].tolist()
    # note maximum number of sentences in a review / words in a sentence
    list_num_sentences = sorted([len(review) for review in data_both], reverse=True)
    max_num_sentences = max(list_num_sentences)
    print('\nMaximum number of sentences in a review: ' + str(max_num_sentences))
    print('Length of 10 longest reviews: ' + str(list_num_sentences[:10]))
    print('Length of 10 shortest reviews: ' + str(list_num_sentences[-10:]))
    print('Set number of sentences in a review after which the data will be truncated: ' + str(ARGS.num_sentences)) 
    list_num_words = sorted([len(sentence) for review in data_both for sentence in review], reverse=True)
    max_num_words = max(list_num_words)
    print('\nMaximum number of words in a sentence: ' + str(max_num_words))
    print('Length of 10 longest sentences: ' + str(list_num_words[:10]))
    print('Length of 10 shortest sentences: ' + str(list_num_words[-10:]))   
    print('Set number of words in a sentence after which the data will be truncated: ' + str(ARGS.num_words))
    # pad reviews to meet desired maximum number of sentences in a review / words in a sentence
    for split in ['data_train', 'data_test']:
        data = splits[split]['codes']
        padded_data = []
        for review in data:
            # TODO: decide if we want to pad/truncate post or pre
            padded_sentences = pad_sequences(review, maxlen=ARGS.num_words, value=ARGS.pad_id,
                                             padding='post', truncating='post').tolist()
            if len(padded_sentences) > 50:
                padded_review = padded_sentences[:50]
            else:
                empty_sentences = [[0] * ARGS.num_words for i in range(ARGS.num_sentences-len(padded_sentences))]
                padded_review = padded_sentences + empty_sentences
            padded_data.append(padded_review)
        splits[split]['codes'] = padded_data
    return splits


def pickle_dictionary(dictionary):
    pd.to_pickle(dictionary, ARGS.path['dictionary'])
    print('\nDictionary size: ' + str(len(dictionary)))
    print('\nFirst ten examples from "dictionary":')
    dictionary_head = {k: dictionary[k] for k in list(dictionary)[:10]}
    print(dictionary_head)


def pickle_splits(splits):
    for split in splits.keys():
        pd.to_pickle(splits[split], ARGS.path[split])
        print('\nData frame for "' + split + '.pkl":')
        print(splits[split].head())


def main(ARGS):
    print('>>> Reading data')
    df = read_data()
    print('\nExample review before any preprocessing:')
    print(df['review'][3])

    print('\n>>> Cleaning data')
    df['codes'] = clean_data(df['review'])
    print('\nExample review after cleaning:')
    print(df['codes'][3])

    print('\n>>> Splitting data into train/test')
    df_splits = split_data(df)

    print('\n>>> Encoding data')
    # print('\nExample review before encoding:')
    # print(df_splits['data_train']['codes'][3])
    df_splits, dictionary = encode_data(df_splits)
    print('\nExample review after encoding:')
    print(df_splits['data_train']['codes'][3])

    print('\n>>> Padding data')
    df_splits = pad_data(df_splits)

    print('\n>>> Pickling dictionary')
    pickle_dictionary(dictionary)

    print('\n>>> Pickling data splits')
    pickle_splits(df_splits)


if __name__ == '__main__':
    # TODO: Uncomment later
    # print('\n>>> Initialize arguments')
    # ARGS = Arguments(dataset='IMDB', dir_data='data', dir_model='model', 
    #                 preprocessing='lemmatize', stopwords='remove',
    #                 num_codes=100000, num_sentences=50, num_words=50,
    #                 emb_size=200, alpha_rec_size=200, beta_rec_size=200, 
    #                 dropout_input=0.0, dropout_context=0.0, l2=0.0,
    #                 epochs=1, batch_size=128)
    # main(ARGS)
    # TODO: Save for all preprocessing options
    # for p in ['lemmatize', 'stem']:
    #     for s in ['remove', 'include']:
    #         print('\n>>> Initialize arguments')
    #         ARGS = Arguments(dataset='IMDB', dir_data='data', dir_model='model', 
    #                         preprocessing=p, stopwords=s,
    #                         num_codes=100000, num_sentences=50, num_words=50,
    #                         emb_size=200, alpha_rec_size=200, beta_rec_size=200, 
    #                         dropout_input=0.0, dropout_context=0.0, l2=0.0,
    #                         epochs=1, batch_size=128)
    #         main(ARGS)