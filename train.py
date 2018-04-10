#! /usr/bin/env python
# ./train.py

"""
~/anaconda3/bin/python train.py
tensorboard --host localhost --port 6006 --logdir summaries/
"""

import numpy as np
import pandas as pd
import os
import time
import math
import yaml
import datetime
import jieba
import jieba.posseg as pseg
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import KFold

import data_helpers
from text_fast import TextFast
from text_dnn import TextDNN
from text_cnn import TextCNN
from text_rnn import TextRNN
from text_birnn import TextBiRNN
from text_rcnn import TextRCNN
from text_han import TextHAN


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("using_nn_type", "textcnn", "The type of neural network type (default: textcnn)")  # fasttext textdnn textcnn textrnn textbirnn textrcnn texthan
tf.flags.DEFINE_string("language_type", "en", "Text language type (default: en)")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("cross_val_folds", 10, "Split the training data to validation with k folds")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of hidden layer units (default: 128)")
tf.flags.DEFINE_integer("hidden_layers", 2, "Number of hidden layers (default: 2)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("rnn_size", 300, "Number of units rnn_size (default: 300)")
tf.flags.DEFINE_integer("num_rnn_layers", 3, "Number of rnn layers (default: 3)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

dataset_name = cfg["datasets"]["default"]
if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_dimension = FLAGS.embedding_dim


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
datasets = None
if dataset_name == "mrpolarity":
    datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                                    cfg["datasets"][dataset_name]["negative_data_file"]["path"])
elif dataset_name == "20newsgroup":
    datasets = data_helpers.get_datasets_20newsgroup(subset="train",
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
elif dataset_name == "localdata":
    datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
elif dataset_name == "financenews":
    datasets = data_helpers.get_datasets_financenews(cfg["datasets"][dataset_name]["path"])
x_text, y = data_helpers.load_data_labels(datasets)

# Build vocabulary
if FLAGS.language_type == 'en':
    max_document_length = max([len(x.split(" ")) for x in x_text])+1
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
elif FLAGS.language_type == 'zh':
    def zh_tokenizer(iterator):
        for value in iterator:
            yield list(jieba.cut(value, cut_all=False))
    max_document_length = max([len(list(jieba.cut(x, cut_all=False))) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=zh_tokenizer)
print("Max document length: {:d}".format(max_document_length))
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/dev set
# TODO: This is crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# kfold = KFold(n_splits=FLAGS.cross_val_folds, shuffle=True, random_state=10)
# for train_index, dev_index in kfold.split(x_shuffled, y_shuffled):
#     x_train, x_dev = x_shuffled[train_index], x_shuffled[dev_index]
#     y_train, y_dev = y_shuffled[train_index], y_shuffled[dev_index]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if FLAGS.using_nn_type == 'fasttext':
            nn = TextFast(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                # embedding_size=FLAGS.embedding_dim,
                embedding_size=embedding_dimension,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.using_nn_type == 'textdnn':
            nn = TextDNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                # embedding_size=FLAGS.embedding_dim,
                embedding_size=embedding_dimension,
                hidden_layers=FLAGS.hidden_layers,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.using_nn_type == 'textcnn':
            nn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                # embedding_size=FLAGS.embedding_dim,
                embedding_size=embedding_dimension,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.using_nn_type == 'textrnn':
            nn = TextRNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                # embedding_size=FLAGS.embedding_dim,
                embedding_size=embedding_dimension,
                rnn_size=FLAGS.rnn_size,
                num_layers=FLAGS.num_rnn_layers,
                # batch_size=FLAGS.batch_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.using_nn_type == 'textbirnn':
            nn = TextBiRNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                # embedding_size=FLAGS.embedding_dim,
                embedding_size=embedding_dimension,
                rnn_size=FLAGS.rnn_size,
                num_layers=FLAGS.num_rnn_layers,
                # batch_size=FLAGS.batch_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.using_nn_type == 'textrcnn':
            nn = TextRCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                # embedding_size=FLAGS.embedding_dim,
                embedding_size=embedding_dimension,
                batch_size=FLAGS.batch_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.using_nn_type == 'texthan':
            nn = TextHAN(
                sequence_length=x_train.shape[1],
                num_sentences=3,
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                # embedding_size=FLAGS.embedding_dim,
                embedding_size=embedding_dimension,
                hidden_size=FLAGS.rnn_size,
                batch_size=FLAGS.batch_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.AdamOptimizer(1e-3)
        optimizer = tf.train.AdamOptimizer(nn.learning_rate)
        # Clip the gradient to avoid larger ones
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(nn.loss, tvars), FLAGS.grad_clip)
        # grads_and_vars = optimizer.compute_gradients(nn.loss)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", nn.loss)
        acc_summary = tf.summary.scalar("accuracy", nn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Initialize the word embedding vectors
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec file has been loaded")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")
            sess.run(nn.W.assign(initW))

        # def train_step(x_batch, y_batch):
        def train_step(x_batch, y_batch, learning_rate):
            """
            A single training step
            """
            feed_dict = {
              nn.input_x: x_batch,
              nn.input_y: y_batch,
              nn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              nn.learning_rate: learning_rate
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, nn.loss, nn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("{}: step {}, loss {:g}, acc {:g}, lr {:g}".format(time_str, step, loss, accuracy, learning_rate))            
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            if FLAGS.using_nn_type in ['fasttext', 'textdnn', 'textcnn', 'textrnn', 'textbirnn']:
                feed_dict = {
                nn.input_x: x_batch,
                nn.input_y: y_batch,
                nn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, nn.loss, nn.accuracy],
                    feed_dict)
            elif FLAGS.using_nn_type in ['textrcnn', 'texthan']:
                loss_sum = 0
                accuracy_sum = 0
                summaries = None
                step = None
                batches_in_dev = len(y_batch) // FLAGS.batch_size
                for batch in range(batches_in_dev):
                    start_index = batch * FLAGS.batch_size
                    end_index = (batch + 1) * FLAGS.batch_size
                    feed_dict = {
                        nn.input_x: x_batch[start_index:end_index],
                        nn.input_y: y_batch[start_index:end_index],
                        nn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, nn.loss, nn.accuracy],
                        feed_dict)
                    loss_sum += loss
                    accuracy_sum += accuracy
                loss = loss_sum / batches_in_dev
                accuracy = accuracy_sum / batches_in_dev
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # It uses dynamic learning rate with a high value at the beginning to speed up the training
        max_learning_rate = 0.005
        min_learning_rate = 0.0001
        decay_speed = FLAGS.decay_coefficient*len(y_train)/FLAGS.batch_size
        # Training loop. For each batch...
        counter = 0
        for batch in batches:
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
            counter += 1
            x_batch, y_batch = zip(*batch)
            # train_step(x_batch, y_batch)
            train_step(x_batch, y_batch, learning_rate)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        # Save config to csv
        attrs = []
        values = []
        for attr, value in sorted(FLAGS.__flags.items()):
            attrs += [attr]
            values += [value]
        info = pd.DataFrame()
        info['attr'] = attrs
        info['value'] = values
        info.to_csv(out_dir + '/config.csv')
