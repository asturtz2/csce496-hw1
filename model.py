import tensorflow as tf

def regularizer():
    return tf.contrib.layers.l2_regularizer(scale=1.0)

def dense_layer(x, size, regularize = True):
    return tf.layers.dense(
        x
        size,
        kernel_regularizer = regularizer() if regularize else None,
        bias_regularizer   = regularizer() if regularize else None,
        activation         = tf.nn.relu,
        name               = layer_name
    )

#Rewrites
def model_1(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = dense_layer(x, 'hidden_layer_1', 256, False)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 128, False)
        return dense_layer(hidden_2, 'output', 10, False)

def model_2(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = dense_layer(x, 'hidden_layer_1', 512, False)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 256, False)
        return dense_layer(hidden_2, 'output', 10, False)

def model_3(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = dense_layer(x, 'hidden_layer_1', 256)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 128)
        return dense_layer(hidden_2, 'output', 10)

def model_4(x):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = dense_layer(x, 'hidden_layer_1', 512)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 256)
        return dense_layer(hidden_2, 'output', 10)

def model_5(x):
    KEEP_PROB=0.7
    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(x, KEEP_PROB)
        hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 256, False)
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 128, False)
        dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
        return dense_layer(dropped_hidden_2, 'output', 10, False)

def model_6(x):
    KEEP_PROB=0.7
    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(x, KEEP_PROB)
        hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 512, False)
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 256, False)
        dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
        return dense_layer(dropped_hidden_2, 'output', 10, False)

def model_7(x):
    KEEP_PROB=0.7
    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(x, KEEP_PROB)
        hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 256)
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 128)
        dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
        return dense_layer(dropped_hidden_2, 'output', 10)

def model_8(x):
    KEEP_PROB=0.7
    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(x, KEEP_PROB)
        hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 512)
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 256)
        dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
        return dense_layer(dropped_hidden_2, 'output', 10)


# def model_1(x):
#     with tf.name_scope('linear_model') as scope:
#         hidden_1 = tf.layers.dense(x,
#              128,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer')
#         output = tf.layers.dense(
#              hidden_1,
#              10,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              name='output')
#         return output

# def model_2(x):
#     with tf.name_scope('linear_model') as scope:
#         hidden_1 = tf.layers.dense(x,
#              128,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer')
#         hidden_2 = tf.layers.dense(
#              hidden_1,
#              128,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer_2')
#         hidden_3 = tf.layers.dense(
#              hidden_2,
#              128,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer_3')
#         output = tf.layers.dense(
#              hidden_3,
#              10,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              name='output')
#         return output

# def model_3(x):
#     with tf.name_scope('linear_model') as scope:
#         hidden_1 = tf.layers.dense(x,
#              512,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer')
#         output = tf.layers.dense(
#              hidden_1,
#              10,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              name='output')
#         return output

# def model_4(x):
#     with tf.name_scope('linear_model') as scope:
#         hidden_1 = tf.layers.dense(x,
#              512,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer')
#         hidden_2 = tf.layers.dense(
#              hidden_1,
#              512,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer_2')
#         hidden_3 = tf.layers.dense(
#              hidden_2,
#              512,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer_3')
#         output = tf.layers.dense(
#              hidden_3,
#              10,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              name='output')
#         return output

# def model_5(x):
#     KEEP_PROB=0.7
#     with tf.name_scope('linear_model') as scope:

#         dropped_input = tf.layers.dropout(x, KEEP_PROB)
#         hidden = tf.layers.dense(dropped_input,
#                              512,
#                              activation=tf.nn.relu,
#                              name='hidden_layer')
#         dropped_hidden = tf.layers.dropout(hidden, KEEP_PROB)
#         output = tf.layers.dense(dropped_hidden,
#                              10,
#                              name='output_layer')
#         return output

# def deep_dropout_model(inputs, keep_prob):
#     with tf.name_scope('linear_model') as scope:
#         dropped_input = tf.layers.Dropout(keep_prob)
#         hidden_1 = dense_layer('hidden_layer_1', size = 256, regularize = False)
#         hidden_2 = dense_layer('hidden_layer_2', size = 256, regularize = False)
#         dropped_hidden = tf.layers.Dropout(keep_prob)
#         output = dense_layer('output', size = 10, regularize = False)
#         return output(dropped_hidden(hidden_2(hidden_1(dropped_input(inputs)))))


# def model_6(x):
#     KEEP_PROB=0.8
#     with tf.name_scope('linear_model') as scope:

#         dropped_input = tf.layers.dropout(x, KEEP_PROB)
#         hidden = tf.layers.dense(dropped_input,
#                              256,
#                              activation=tf.nn.relu,
#                              name='hidden_layer')
#         hidden2 = tf.layers.dense(hidden,
#                              256,
#                              activation=tf.nn.relu,
#                              name='hidden_layer2')
#         dropped_hidden = tf.layers.dropout(hidden2, KEEP_PROB)
#         output = tf.layers.dense(dropped_hidden,
#                              10,
#                              name='output_layer')
#         return output
