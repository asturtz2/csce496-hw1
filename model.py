import tensorflow as tf
def layers(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x,
             128,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer')
        output = tf.layers.dense(
             hidden_1,
             10,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             name='output')
        return output

def layers2(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x,
             128,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer')
        hidden_2 = tf.layers.dense(
             hidden_1,
             128,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer_2')
        hidden_3 = tf.layers.dense(
             hidden_2,
             128,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer_3')
        output = tf.layers.dense(
             hidden_3,
             10,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             name='output')
        return output

def layers3(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x,
             512,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer')
        output = tf.layers.dense(
             hidden_1,
             10,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             name='output')
        return output

def layers4(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x,
             512,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer')
        hidden_2 = tf.layers.dense(
             hidden_1,
             512,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer_2')
        hidden_3 = tf.layers.dense(
             hidden_2,
             512,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             activation=tf.nn.relu,
             name='hidden_layer_3')
        output = tf.layers.dense(
             hidden_3,
             10,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             name='output')
        return output
def layers5(x):
    KEEP_PROB=0.7
    with tf.name_scope('linear_model') as scope:
        
        dropped_input = tf.layers.dropout(x, KEEP_PROB)
        hidden = tf.layers.dense(dropped_input,
                             512,
                             activation=tf.nn.relu,
                             name='hidden_layer')
        dropped_hidden = tf.layers.dropout(hidden, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden,
                             10,
                             name='output_layer')
        return output
def layers6(x):
    KEEP_PROB=0.8
    with tf.name_scope('linear_model') as scope:
        
        dropped_input = tf.layers.dropout(x, KEEP_PROB)
        hidden = tf.layers.dense(dropped_input,
                             256,
                             activation=tf.nn.relu,
                             name='hidden_layer')
        hidden2 = tf.layers.dense(hidden,
                             256,
                             activation=tf.nn.relu,
                             name='hidden_layer2')
        dropped_hidden = tf.layers.dropout(hidden2, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden,
                             10,
                             name='output_layer')
        return output