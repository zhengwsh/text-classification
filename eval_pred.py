#! /usr/bin/env python
# -*- coding: UTF-8 -*-

"""
~/anaconda3/bin/python eval_pred_news.py --evaluate --checkpoint_dir="./runs/1523240176/checkpoints/"
~/anaconda3/bin/python eval_pred_news.py --predict --checkpoint_dir="./runs/1523240176/checkpoints/"
"""

import numpy as np
import pandas as pd
import os
import time
import csv
import yaml
import datetime

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics
import jieba
import jieba.posseg as pseg

import data_helpers


def zh_tokenizer(iterator):
    for value in iterator:
        yield list(jieba.cut(value, cut_all=False))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("evaluate", False, "Evaluate on all training data")
tf.flags.DEFINE_boolean("predict", False, "Predict on test dataset")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# CHANGE THIS: Load data. Load your own data here
datasets = None
dataset_name = cfg["datasets"]["default"]
if FLAGS.evaluate:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                              categories=cfg["datasets"][dataset_name]["categories"],
                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "financenews":
        datasets = data_helpers.get_datasets_financenews(cfg["datasets"][dataset_name]["path"])
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)

elif FLAGS.predict:
    if dataset_name == "mrpolarity":
        datasets = {"target_names": ['positive_examples', 'negative_examples']}
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = None
    elif dataset_name == "20newsgroup":
        datasets = {"target_names": ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']}
        x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
                 "I am in the market for a 24-bit graphics card for a PC"]
        y_test = None
    elif dataset_name == "financenews":
        datasets = {"target_names": ['strong_neg_examples', 'weak_neg_examples', 'neutral_examples', 'weak_pos_examples', 'strong_pos_examples']}        
        datasets = data_helpers.get_datasets_financenews_test(cfg["datasets"][dataset_name]["test_path"])
        x_raw = data_helpers.load_data(datasets)
        y_test = None
        # datasets = {"target_names": ['strong_neg_examples', 'weak_neg_examples', 'neutral_examples', 'weak_pos_examples', 'strong_pos_examples']}
        # x_raw = ["这是什么垃圾股票", "我赚翻了"]
        # y_test = None
    

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nPredicting...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]
 
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for index, x_test_batch in enumerate(batches):
            # batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            # all_predictions = np.concatenate([all_predictions, batch_predictions])
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}".format(time_str, (index+1)*FLAGS.batch_size))

# Print accuracy if y_test is defined
if y_test is not None:
    y_test = y_test[:len(y_test)-len(y_test)%FLAGS.batch_size]    
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print(metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
x_raw = x_raw[:len(x_raw)-len(x_raw)%FLAGS.batch_size]
predictions_human_readable = np.column_stack((np.array(x_raw),
                                              [int(prediction)+1 for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities]))
predict_results = pd.DataFrame(predictions_human_readable, columns=['Content','Label','Probabilities'])
if FLAGS.evaluate:
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "evaluation.csv")
elif FLAGS.predict:
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")    
print("Saving evaluation to {0}".format(out_path))
predict_results.to_csv(out_path)
# with open(out_path, 'w') as f:
    # csv.writer(f).writerows(predictions_human_readable)
