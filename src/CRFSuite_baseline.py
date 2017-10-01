from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import numpy as np
import glob
import torch
import re

from train import load_wordvecs

GLOVE_PATH = '../InferSent/dataset/GloVe/glove.840B.300d.txt' # for sentence embeddings
# WORD_VECTORS = '../../word_vectors/glove.6B.100d.txt'
LABELS = ["A", "ATV", "ATO", "DO", "D", "NSTO", "QB", "QS", "QR", "OO", "O"]
MAX_SEQ_LEN = 400 # number of sentences in a conversation file
SENTENCE_ENCODING_DIM = 4096
MAX_SENTENCE_LEN = 30 # number of tokens in a sentence
WORD_EMBEDDING_DIM = 100

# word_vectors = load_wordvecs()

def get_embedding_features(sentence_vectors, speakers):
    """convert 2D numpy array to list of dictionaries"""
    feature_list = []
    N = len(sentence_vectors)
    for i, vector in enumerate(sentence_vectors):
        f = {}
        # if i == 0:
        #     f['START'] = True
        # if i == N-1:
        #     f['END'] = True
        # f['length'] = sentence_lengths[i]
        # f['time_step'] = i
        if i > 0 and speakers[i] == speakers[i-1]:
            f['same_speaker'] = 1
        else:
            f['same_speaker'] = 0
        f['emb'] = {}
        for j, x in enumerate(vector):
            f['emb']['emb'+str(j)] = x
        if i > 0:
            f['-1:emb'] = feature_list[i-1]['emb']
            feature_list[i-1]['+1:emb'] = f['emb']

        feature_list.append(f)

    return feature_list


def read_data(file_path, file_list=None):
    infersent = torch.load('../InferSent/encoder/infersent.allnli.pickle')
    infersent.set_glove_path(GLOVE_PATH)
    infersent.build_vocab_k_words(K=1000)
    speaker_regx = re.compile(r'[A-Z]+:')

    if file_list is None:
        input_files = glob.glob(file_path+'*.dat')
    else:
        input_files = [file_path+x.strip() for x in open(file_list)]

    X_speakers = []
    X_sentences = []
    Y_labels = []
    for input_file in input_files:
        speakers = []
        sentences = []
        # sentence_lengths = []
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
                    content = content[match.end() + 1:]

                sentences.append(content)
                speakers.append(speaker) # record the identity of the speaker
                # sentence_lengths.append(len(nltk.word_tokenize(content)))

                if label == 'O':
                    labels.append(label)
                elif label[1] == '-':
                    labels.append(label[2:])
                else:
                    print("Wrong label:", label)
                    continue

        infersent.update_vocab(sentences, tokenize=True)
        sentence_vectors = infersent.encode(sentences, tokenize=True)
        x_sentences = get_embedding_features(sentence_vectors, speakers) #list of dictionaries
        print("conversation length", len(x_sentences))
        X_sentences.append(x_sentences)

        Y_labels.append(labels)

    return X_sentences, Y_labels

def train(training_path, model_dest, valid_path=None):
    X_train, y_train = read_data(training_path)
    if valid_path:
        X_valid, y_valid = read_data(valid_path)

    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train(model_dest + 'model.crfsuite')
    print("finished training!")

    if valid_path:
        tagger = pycrfsuite.Tagger()
        tagger.open(model_dest + 'model.crfsuite')
        y_pred = [tagger.tag(xseq) for xseq in X_valid]
        bio_classification_report(y_valid, y_pred)

def train_cv(data_path, model_dest, file_list=None):
    X, y = read_data(data_path, file_list=file_list)
    roll_num = round(len(y)/5)
    prf_scores = []
    accuracies = []

    for cv in range(5):
        X_test = X[0:roll_num] # [[{},...],...]
        y_test = y[0:roll_num]
        X_train = X[roll_num:]
        y_train = y[roll_num:]
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 200,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        model_name = 'model.crfsuite' + str(cv)
        trainer.train(model_dest + model_name)
        print("finished training %s!" % model_name)

        tagger = pycrfsuite.Tagger()
        tagger.open(model_dest + model_name)
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        # print("Prediction accuracy:", accuracy_score(y_valid, y_pred))
        report = bio_classification_report(y_test, y_pred)
        accuracies.append(report[0])
        prf_scores.append(report[1][0:3])

        X = np.roll(X, roll_num, axis=0)
        y = np.roll(y, roll_num, axis=0)

    mean_accuracy = np.mean(accuracies)
    mean_prf_scores = np.mean(np.array(prf_scores), axis=0)
    print("Accuracies:", accuracies)
    print("Average accuracy of 5-folds cv: %f.3" % mean_accuracy)
    print("\nPrecision, recall, f1 scores:")
    for item in prf_scores:
        print(item)
    print("Average precision, recall, f1:", mean_prf_scores)


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    # confusion = np.zeros((10, 10), dtype='int16')
    # for true, pred in zip(y_true_combined, y_pred_combined):
    #     # build confusion matrix
    #     try:
    #         confusion[true, pred] += 1
    #     except IndexError:
    #         confusion[true, class_indices['OO']] += 1  # put it in 'OO'
    # print("confusion matrix")
    # print("rows: actual labels.  columns: predicted labels.")
    # for i, row in enumerate(confusion):
    #     print(LABELS[i], ": ", row)

    print(classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        digits=3
    ))
    return accuracy_score(y_true_combined, y_pred_combined), \
           precision_recall_fscore_support(y_true_combined, y_pred_combined, average='weighted')

if __name__ == '__main__':
    # training_path = '../data/training/'
    # valid_path = '../data/test/'
    # # training_path = '../data/exp/*.dat'#quick experiment
    # model_dest = '../models/model.crfsuite'
    #
    # train(training_path, model_dest, valid_path=valid_path)
    # # train_batch(training_path, model_dest, valid_path=valid_path)

    data_path = '../data/all/'
    model_dest = '../models/'
    file_list = '../data/randomized_list.txt'
    train_cv(data_path, model_dest, file_list=file_list)