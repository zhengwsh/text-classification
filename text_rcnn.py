import tensorflow as tf
import numpy as np
import copy


class TextRCNN(object):
    """
    A RNN-CNN for text classification/regression.
    Uses an embedding layer, followed by a recurrent, convolutional, fully-connected (and softmax) layer.
    """
    def __init__(
      self, model_type, sequence_length, num_classes, vocab_size,
      embedding_size, batch_size, l2_reg_lambda=0.5):  # batch_size, 

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # When trainable parameter equals True the embedding vector is non-static, otherwise is static
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W", trainable=True)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # [None, sequence_length, embedding_size]

        # Create a recurrent-convolutional layer for each rnn layer
        with tf.name_scope('rcnn'):
            # define weights here
            self.initializer = tf.random_normal_initializer(stddev=0.1)
            self.left_side_first_word = tf.get_variable("left_side_first_word", shape=[batch_size, embedding_size], initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word", shape=[batch_size, embedding_size], initializer=self.initializer)
            self.W_l = tf.get_variable("W_l", shape=[embedding_size, embedding_size], initializer=self.initializer)
            self.W_r = tf.get_variable("W_r", shape=[embedding_size, embedding_size], initializer=self.initializer)
            self.W_sl = tf.get_variable("W_sl", shape=[embedding_size, embedding_size], initializer=self.initializer)
            self.W_sr = tf.get_variable("W_sr", shape=[embedding_size, embedding_size], initializer=self.initializer)
            
            # rnn-cnn layer
            def get_context_left(context_left, embedding_previous):
                left_c = tf.matmul(context_left, self.W_l)  #context_left:[batch_size,embed_size]; W_l:[embed_size,embed_size]
                left_e = tf.matmul(embedding_previous, self.W_sl)  #embedding_previous; [batch_size,embed_size]
                left_h = left_c + left_e
                context_left = tf.nn.relu(left_h, name="relu") # [None,embed_size]
                return context_left
            def get_context_right(context_right, embedding_afterward):
                right_c = tf.matmul(context_right, self.W_r)
                right_e = tf.matmul(embedding_afterward, self.W_sr)
                right_h = right_c + right_e
                context_right = tf.nn.relu(right_h, name="relu")
                return context_right

            #1. get splitted list of word embeddings
            #2. get list of context left
            embedded_words_split = tf.split(self.embedded_chars, sequence_length, axis=1) #sentence_length * [None,1,embed_size]
            embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split] #sentence_length * [None,embed_size]
            embedding_previous = self.left_side_first_word
            context_left_previous = tf.zeros((batch_size, embedding_size))
            context_left_list=[]
            for i, current_embedding_word in enumerate(embedded_words_squeezed): #sentence_length * [None,embed_size]
                context_left = get_context_left(context_left_previous, embedding_previous) #[None,embed_size]
                context_left_list.append(context_left) #append result to list
                embedding_previous = current_embedding_word #assign embedding_previous
                context_left_previous = context_left #assign context_left_previous
            #3. get context right
            embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
            embedded_words_squeezed2.reverse()
            embedding_afterward = self.right_side_last_word
            context_right_afterward = tf.zeros((batch_size, embedding_size))
            context_right_list=[]
            for j, current_embedding_word in enumerate(embedded_words_squeezed2):
                context_right = get_context_right(context_right_afterward, embedding_afterward)
                context_right_list.append(context_right)
                embedding_afterward = current_embedding_word
                context_right_afterward = context_right
            #4.ensemble left, embedding, right to output
            output_list=[]
            for index, current_embedding_word in enumerate(embedded_words_squeezed):
                representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]], axis=1)
                output_list.append(representation) #shape:sentence_length * [None,embed_size*3]
            #5. stack list to a tensor
            outputs = tf.stack(output_list, axis=1) #shape:[None,sentence_length,embed_size*3]
            #6. max pooling
            self.output = tf.reduce_max(outputs, axis=1) #shape:[None,embed_size*3]

        # Add dropout
        with tf.name_scope("dropout"):
            self.rnn_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size*3, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.rnn_drop, W, b, name="scores")
            if model_type == 'clf':
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
            elif model_type == 'reg':
                self.predictions = tf.reduce_max(self.scores, 1, name="predictions")
                self.predictions = tf.expand_dims(self.predictions, -1)

        # Calculate mean cross-entropy loss, or root-mean-square error loss
        with tf.name_scope("loss"):
            if model_type == 'clf':
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            elif model_type == 'reg':
                losses = tf.sqrt(tf.losses.mean_squared_error(predictions=self.predictions, labels=self.input_y))
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            if model_type == 'clf':
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            elif model_type == 'reg':
                self.accuracy = tf.constant(0.0, name="accuracy")