
def layers(input):
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.Dense(
             128,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
             activation=tf.nn.relu,
             name='hidden_layer_1'
        )
        hidden_2 = tf.layers.Dense(
             100,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
             activation=tf.nn.relu,
             name='hidden_layer_2'
        )
        output = tf.layers.Dense(
             10,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
             name='output'
        )
        return output(hidden_2(hidden_1(input)))

