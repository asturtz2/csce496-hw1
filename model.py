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
             name='hidden_layer_2')
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
             name='hidden_layer_2')
        output = tf.layers.dense(
             hidden_3,
             10,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
             name='output')
        return output