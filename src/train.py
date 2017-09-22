from __future__ import print_function
import config
import re
import glob
from string import ascii_uppercase
import numpy as np
import copy
import random
import torch
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, LSTM, Dense, Merge, Concatenate, MaxPooling1D, TimeDistributed, Flatten, Masking, Input, Dropout, Bidirectional
from keras.layers import concatenate, SimpleRNN, GRU, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json, load_model
from sklearn.metrics import classification_report, accuracy_score
import nltk
from keras_contrib.layers import CRF
import keras.backend as K

GLOVE_PATH = '../InferSent/dataset/GloVe/glove.840B.300d.txt' # for sentence embeddings
# WORD_VECTORS = '../../word_vectors/glove.6B.100d.txt'
LABELS = ["A", "ATO", "ATV", "D", "DO", "NSTO", "OO", "QB", "QR", "QS",  "O"]
MAX_SEQ_LEN = 400 # number of sentences in a conversation file
SENTENCE_ENCODING_DIM = 4096
MAX_SENTENCE_LEN = 30 # number of tokens in a sentence
WORD_EMBEDDING_DIM = 300
CUSTOM_OBJECTS={"CRF": CRF, 'CRFLoss': CRF.loss_function}

def load_wordvecs():
    print("loading word vectors...")
    embeddings_index = {}
    with open(GLOVE_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print("not included in word vectors:", values[:-300])
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def get_word_embeddings(text, word_vectors):
    """Given a text, tokenize it and return the word embeddings of the token list"""
    tokens = nltk.word_tokenize(text)
    embeddings = np.array([word_vectors.get(x, np.zeros(WORD_EMBEDDING_DIM)) for x in tokens]) # (tokens, dim)
    embeddings = np.expand_dims(embeddings, axis=0) # (sentence, tokens, dim), only 1 sentence here
    embeddings = pad_sequences(embeddings, maxlen=MAX_SENTENCE_LEN, dtype='float32', padding='pre', truncating='post', value=-1.)
    return embeddings # (1, MAX_SENTENCE_LEN, dim)

def get_average_embeddings(text, word_vectors):
    """Given a text, tokenize it and return the word embeddings of the token list"""
    tokens = nltk.word_tokenize(text)
    embeddings = np.array([word_vectors.get(x, np.zeros(WORD_EMBEDDING_DIM)) for x in tokens]) # (tokens, dim)
    m_embedding = np.mean(embeddings, axis=0) # (dim,)
    return m_embedding

def get_end_embeddings(text, word_vectors):
    """Given a text, tokenize it and return the concatenated word embeddings of the first and last tokens"""
    tokens = nltk.word_tokenize(text)
    first = word_vectors.get(tokens[0], np.zeros(WORD_EMBEDDING_DIM))
    last = word_vectors.get(tokens[-1], np.zeros(WORD_EMBEDDING_DIM))
    embeddings = first + last
    return embeddings

def read_data(file_path, word_vectors, padding=True):
    one_hot = {}
    for i, l in enumerate(ascii_uppercase):
        bits = np.zeros(26)
        bits[i] = 1
        one_hot[l] = bits

    infersent = torch.load('../InferSent/encoder/infersent.allnli.pickle')
    infersent.set_glove_path(GLOVE_PATH)
    infersent.build_vocab_k_words(K=1000)
    speaker_regx = re.compile(r'[A-Z]+:')

    input_files = glob.glob(file_path)
    X_speakers = []
    X_sentences = []
    X_word_embeddings = []
    X_sentence_lengths = []
    Y_labels = []
    for input_file in input_files:
        speakers = []
        sentences = []
        word_embeddings = []
        sentence_lengths = []
        labels = []
        with open(input_file) as f:
            speaker = ''
            for line in f:
                try:
                    content, label = line.strip().split('\t')
                except:
                    print("no label:", content)
                    continue
                if label == 'O': # speaker is not a team member, or utterance unrecognizable
                    continue

                match = speaker_regx.match(line)
                if match:
                    speaker = match.group()[:-1]
                    content = content[match.end()+1:]

                speakers.append(one_hot.get(speaker, np.zeros(26)))
                sentences.append(content)
                word_embeddings.append(get_average_embeddings(content, word_vectors))
                sentence_lengths.append(len(nltk.word_tokenize(content)))
                # if word_embeddings is None:
                #     word_embeddings = get_end_embeddings(content, word_vectors)
                # else:
                #     word_embeddings = np.concatenate((word_embeddings, get_end_embeddings(content, word_vectors)), axis=0)

                if label =='O':
                    labels.append(label)
                elif label[1] == '-':
                    labels.append(label[2:])
                else:
                    print("Wrong label:", label)
                    continue
                    # labels.append('O')

        x_speakers = np.array(speakers)
        X_speakers.append(x_speakers)

        print("# sentencees", input_file, len(sentences))

        infersent.update_vocab(sentences, tokenize=True)
        x_sentences = infersent.encode(sentences, tokenize=True)
        X_sentences.append(x_sentences)

        X_word_embeddings.append(word_embeddings)

        sentence_lengths = np.expand_dims(np.array(sentence_lengths), axis=-1) # in order to make 3D tensor
        X_sentence_lengths.append(sentence_lengths)

        y_labels = label_to_int(labels)
        Y_labels.append(y_labels)

    if padding:
        X_speakers = pad_sequences(X_speakers, maxlen=MAX_SEQ_LEN, dtype='int32', padding='post', truncating='post', value=-1)
        X_sentences = pad_sequences(X_sentences, maxlen=MAX_SEQ_LEN, dtype='float32', padding='post', truncating='post',
                                   value=-1.)
        X_word_embeddings = pad_sequences(X_word_embeddings, maxlen=MAX_SEQ_LEN, dtype='float32', padding='post', truncating='post',
                                    value=-1.)
        X_sentence_lengths = pad_sequences(X_sentence_lengths, maxlen=MAX_SEQ_LEN, dtype='int32', padding='post', truncating='post',
                                          value=-1.)
        time_steps = np.expand_dims(np.arange(MAX_SEQ_LEN)/MAX_SEQ_LEN, axis=0)
        X_time_steps = np.repeat(time_steps, len(Y_labels), axis=0)
        X_time_steps = np.expand_dims(X_time_steps, axis=-1) # (n, MAX_SEQ_LEN, 1)

        Y_labels = pad_sequences(Y_labels, maxlen=MAX_SEQ_LEN, dtype='int32', padding='post', truncating='post',
                                   value=LABELS.index('O')) # Add a dummy class for padded time steps

    return [X_sentences, X_time_steps], Y_labels
    # return X_sentences, Y_labels


def label_to_int(label_list):
    return np.array([LABELS.index(l) if l in LABELS else LABELS.index('O') for l in label_list])


# def seq2seq():
#     n_classes = len(LABELS)
#
#     # speaker = Sequential()
#     # speaker.add(Masking(mask_value=-1, input_shape=(None, 26)))
#     # speaker.add(Bidirectional(LSTM(2, return_sequences=True)))
#
#     converse = Sequential()
#     converse.add(Masking(mask_value=-1., input_shape=(None, SENTENCE_ENCODING_DIM)))
#     converse.add(Dropout(0.5))
#     converse.add(Bidirectional(LSTM(1024, return_sequences=True)))
#     # converse.add(Bidirectional(LSTM(1024, return_sequences=True)))
#     converse.add(Dropout(0.5))
#
#     # words = Sequential()
#     # words.add(Dropout(0.5, input_shape=(None, 2*WORD_EMBEDDING_DIM)))
#
#     lengths = Sequential()
#     lengths.add(Masking(mask_value=-1., input_shape=(None, 1)))
#
#     model = Sequential()
#     # model.add(Merge([speaker, converse, words], mode='concat'))
#     # model.add(Merge([converse, lengths], mode='concat'))
#     model.add(Concatenate(axis=-1)([converse, lengths]))
#
#     print("merged outpout shape", model.output_shape)
#
#     model.add(TimeDistributed(Dense(512, activation='relu')))
#     model.add(Dropout(0.5))
#     # model.add(TimeDistributed(Dense(256, activation='relu')))
#     # model.add(Dropout(0.5))
#     model.add(Dense(n_classes, activation='softmax'))
#     print("time distributed", model.input_shape, model.output_shape)
#     model.summary()
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['accuracy'])
#     return model

def seq2seq():
    n_classes = len(LABELS)

    converse_input = Input(shape=(None, SENTENCE_ENCODING_DIM))
    # length_input = Input(shape=(None, 1))
    # word_input = Input(shape=(None, WORD_EMBEDDING_DIM))
    time_input = Input(shape=(None, 1))

    converse = Masking(mask_value=-1.)(converse_input)
    converse = Dropout(0.2)(converse)
    converse = Bidirectional(LSTM(1024, return_sequences=True))(converse)
    converse = Bidirectional(LSTM(1024, return_sequences=True))(converse)
    converse = Dropout(0.3)(converse)

    # lengths = Masking(mask_value=-1)(length_input)

    # words = Masking(mask_value=-1.)(word_input)
    # words = Dropout(0.2)(words)

    model = concatenate([converse, time_input], axis=-1)

    # print("merged outpout shape", model.output_shape)

    model = TimeDistributed(Dense(1024, activation='relu'))(model)
    model = Dropout(0.3)(model)
    model = TimeDistributed(Dense(512, activation='relu'))(model)
    model = Dropout(0.3)(model)
    # predictions = TimeDistributed(Dense(n_classes, activation='softmax'))(model)

    crf = CRF(n_classes, sparse_target=True)
    predictions = crf(model)

    model = Model(inputs=[converse_input, time_input], outputs=predictions)
    model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def train(training_path, model_dest, valid_path=None):

    word_vectors = load_wordvecs()

    X, Y_labels = read_data(training_path, word_vectors)
    # Y = []
    # for labels in Y_labels:
    #     Y.append(to_categorical(labels))
    # Y = np.array(Y)
    Y_labels = np.expand_dims(np.array(Y_labels), axis=-1)  # in order to make 3D tensor

    if valid_path is not None:
        Xv, Yv_labels = read_data(valid_path, word_vectors)
        # Yv = []
        # for labels in Yv_labels:
        #     Yv.append(to_categorical(labels))
        # Yv = np.array(Yv)
        Yv_labels = np.expand_dims(np.array(Yv_labels), axis=-1)  # in order to make 3D tensor
        valid_data = (Xv, Yv_labels)
        # valid_data = (Xv, Yv)
    else:
        valid_data = None

    model = seq2seq()
    architecture = model.to_json()
    with open(model_dest + 'arch.json', "w") as f:
        f.write(architecture)

    # earlystopping = EarlyStopping(monitor='crf_1_acc_1', patience=30, verbose=0, mode='auto')
    earlystopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(model_dest+'weights.h5', monitor='val_acc', save_best_only=True,
                                 save_weights_only=True)
    model.fit(X, Y_labels, validation_split=0.2, validation_data=valid_data,
              callbacks=[earlystopping, checkpoint], shuffle=True, batch_size=4, epochs=300)

    # best_model = model_from_json(open(model_dest + 'arch.json').read(), custom_objects=CUSTOM_OBJECTS)
    # best_model.load_weights(model_dest + 'weights.h5')
    # evaluate(best_model, valid_data, results_dest='../predictions/')

    evaluate(model, valid_data, results_dest='../predictions/')
    model.save_weights(model_dest + 'final_weights.h5')


# def train_batch(training_path, model_dest, valid_path=None):
#     """This allows variant sequence length. Only batch size 1 is used."""
#
#     X_speakers, X_sentences, Y_labels = read_data(training_path, padding=False)
#
#     Y = []
#     for labels in Y_labels:
#         Y.append(to_categorical(labels))
#     # Y = np.array(Y)
#
#     if valid_path is not None:
#         Xv_speakers, Xv_sentences, Yv_labels = read_data(valid_path)
#         Yv = []
#         for labels in Yv_labels:
#             Yv.append(to_categorical(labels))
#         # Yv = np.array(Yv)
#         valid_data = ([Xv_speakers, Xv_sentences], Yv)
#     else:
#         valid_data = None
#
#     model = seq2seq()
#     earlystopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')
#     zipped_data = list(zip(X_speakers, X_sentences, Y))
#
#     for epochs in range(100):
#         random.shuffle(zipped_data)
#         accuma_metrics = [0, 0]
#         n = 0
#         for x0, x1, y in zipped_data:
#             # print(y)
#             metrics = model.train_on_batch([np.expand_dims(x0, axis=0), np.expand_dims(x1, axis=0)], np.expand_dims(y, axis=0))
#             accuma_metrics[0] += metrics[0]
#             accuma_metrics[1] += metrics[1]
#             n += 1
#             model.reset_states()
#         print("\nepoch # %d completed. loss %.4f, acc %.4f" % (epochs+1, accuma_metrics[0]/n, accuma_metrics[1]/n))
#
#         print("evaluate on validation set...")
#         accuma_val_metrics = [0, 0]
#         n = 0
#         for x0, x1, y in zip(Xv_speakers, Xv_sentences, Yv):
#             val_metrics = model.test_on_batch([np.expand_dims(x0, axis=0), np.expand_dims(x1, axis=0)], np.expand_dims(y, axis=0))
#             accuma_val_metrics[0] += val_metrics[0]
#             accuma_val_metrics[1] += val_metrics[1]
#             n += 1
#         print("val_loss %.4f, val_acc %.4f" % (accuma_val_metrics[0]/n, accuma_val_metrics[1]/n))
#
#     evaluate(model, valid_data, results_dest='../predictions/')
#     model.save(model_dest)


def evaluate(model, test_data, results_dest=None):

    y_predict = model.predict(test_data[0])
    y_predict = np.argmax(y_predict, axis=-1)
    labels_predict = y_predict.reshape(-1)
    print("predictions", labels_predict.shape)

    # labels_true = np.argmax(test_data[1], axis=-1)
    labels_true = test_data[1]
    labels_true = labels_true.reshape(-1)

    used_indexes = np.where(labels_true < LABELS.index('O'))[0]
    labels_true = labels_true[used_indexes]
    labels_predict = labels_predict[used_indexes]

    confusion = np.zeros((10, 10), dtype='int16')

    for true, pred in zip(labels_true, labels_predict):
        # build confusion matrix
        try:
            confusion[true, pred] += 1
        except IndexError:
            confusion[true, 6] += 1 # put it in 'OO'
    print("confusion matrix")

    print("rows: actual labels.  columns: predicted labels.")
    for i, row in enumerate(confusion):
        print(i, ": ", row)

    print(classification_report(labels_true, labels_predict, labels=range(0,10), target_names=LABELS[0:10], digits=3))
    print("accuracy", accuracy_score(labels_true, labels_predict))

    if results_dest:
        for i, y in enumerate(y_predict):
            with open(results_dest+str(i)+'.dat', 'w') as f:
                f.write('\n'.join([LABELS[s] for s in y]))


if __name__ == '__main__':
    training_path = '../data/training/*.dat'
    valid_path = '../data/test/*.dat'
    # training_path = '../data/exp/*.dat'#quick experiment
    model_dest = '../models/'

    train(training_path, model_dest, valid_path=valid_path)
    # train_batch(training_path, model_dest, valid_path=valid_path)

