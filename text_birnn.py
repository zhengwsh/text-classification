import tensorflow as tf
import numpy as np


class TextBiRNN(object):
    """
    A Bi-directional RNN for text classification.
    Uses an embedding layer, followed by a bi-directional recurrent, fully-connected and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, rnn_size, num_layers, l2_reg_lambda=0.5, model='lstm'):  # batch_size, 

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # [None, sequence_length, embedding_size]

        # Create a bi-directional recurrent layer for each rnn layer
        with tf.name_scope('bi'+model):
            if model == 'rnn':
                cell_fun = tf.nn.rnn_cell.BasicRNNCell
            elif model == 'gru':
                cell_fun = tf.nn.rnn_cell.GRUCell
            elif model == 'lstm':
                cell_fun = tf.nn.rnn_cell.BasicLSTMCell
            
            def get_bi_cell():
                fw_cell = cell_fun(rnn_size, state_is_tuple=True) #forward direction cell
                bw_cell = cell_fun(rnn_size, state_is_tuple=True) #backward direction cell
                # fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
                # bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
                return fw_cell, bw_cell
            
            # Stacking multi-layers
            # cell = tf.nn.rnn_cell.MultiRNNCell([get_bi_cell() for _ in range(num_layers)])
            # initial_state = cell.zero_state(None, tf.float32)

            # Bi-lstm layer
            fw_cell, bw_cell = get_bi_cell()
            outputs, last_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_chars, dtype=tf.float32)
            # outputs, last_state = tf.nn.dynamic_rnn(cell, self.embedded_chars, dtype=tf.float32)  # , initial_state=initial_state
            # --'outputs' is a tensor of shape [batch_size, max_time, cell_state_size], [batch_size, max_time, cell_state_size]
            # --'last_state' is a tensor of shape [batch_size, cell_state_size], [batch_size, cell_state_size]
            outputs = tf.concat(outputs, axis=2)
            # self.output = outputs[:, -1, :]
            self.output = tf.reduce_mean(outputs, axis=1)
            # self.output = tf.reshape(outputs, [batch_size, -1])

        # Add dropout
        with tf.name_scope("dropout"):
            self.rnn_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[rnn_size*2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.rnn_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")