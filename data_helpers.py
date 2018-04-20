import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    num_batches_per_epoch = data_size // batch_size
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # if end_index - start_index != batch_size:
                # yield shuffled_data[end_index-batch_size:end_index]
            yield shuffled_data[start_index:end_index]


def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
    return datasets


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                       encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


def get_datasets_financenews(data_file):
    """
    Loads finance news data from files, splits the data into sentences and generates labels.
    Returns split sentences and labels.
    """
    df_data = pd.read_csv(data_file)
    datasets = dict()

    use_text = 'Title' # Abstract
    strong_neg_examples = list(df_data[df_data['score']==1][use_text].values)
    strong_neg_examples = [str(s).strip() for s in strong_neg_examples]
    weak_neg_examples = list(df_data[df_data['score']==2][use_text].values)
    weak_neg_examples = [str(s).strip() for s in weak_neg_examples]
    neutral_examples = list(df_data[df_data['score']==3][use_text].values)
    neutral_examples = [str(s).strip() for s in neutral_examples]
    weak_pos_examples = list(df_data[df_data['score']==4][use_text].values)
    weak_pos_examples = [str(s).strip() for s in weak_pos_examples]
    strong_pos_examples = list(df_data[df_data['score']==5][use_text].values)
    strong_pos_examples = [str(s).strip() for s in strong_pos_examples]
    datasets['data'] = strong_neg_examples + weak_neg_examples + neutral_examples + weak_pos_examples + strong_pos_examples

    target = [0 for x in strong_neg_examples] + [1 for x in weak_neg_examples] + [2 for x in neutral_examples] + \
             [3 for x in weak_pos_examples] + [4 for x in strong_pos_examples]
    datasets['target'] = target
    datasets['target_names'] = ['strong_neg_examples', 'weak_neg_examples', 'neutral_examples', 'weak_pos_examples', 'strong_pos_examples']
    return datasets


def get_datasets_financenews_test(data_file):
    """
    Loads finance news data from files, splits the data into sentences.
    Returns split sentences.
    """
    df_data = pd.read_csv(data_file)

    use_text = 'Title' # Abstract
    examples = list(df_data[use_text].values)
    examples = [str(s).strip() for s in examples]

    datasets = dict()
    datasets['data'] = examples
    return datasets


def get_datasets_scoringdocuments(data_file):
    """
    Loads scored documents data from files, splits the data into sentences and generates labels.
    Returns split sentences and score label.
    """
    df_data = pd.read_csv(data_file)
    datasets = dict()

    use_text = 'Abstract'
    examples = list(df_data[use_text].values)
    examples = [str(s).strip() for s in examples]
    datasets['data'] = examples
    target = list(df_data['Score'].values)
    datasets['target'] = target
    datasets['target_names'] = ['document_score']
    return datasets


def get_datasets_scoringdocuments_test(data_file):
    """
    Loads document data from files, splits the data into sentences.
    Returns split sentences.
    """
    df_data = pd.read_csv(data_file)

    use_text = 'Abstract'
    examples = list(df_data[use_text].values)
    examples = [str(s).strip() for s in examples]

    datasets = dict()
    datasets['data'] = examples
    return datasets


def load_data_label(datasets):
    """
    Load data and label
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [sent for sent in x_text]
    # Generate regressor label
    label = []
    for i in range(len(x_text)):
        score = datasets['target'][i]
        label.append([score])
    y = np.array(label)
    return [x_text, y]


def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [sent for sent in x_text]
    # Generate labels (one-hot encoding)
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_data(datasets):
    """
    Load data without labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [sent for sent in x_text]
    return x_text


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors