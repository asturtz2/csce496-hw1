import tensorflow as tf
def layers(x):
	with tf.name_scope('linear_model') as scope:
		hidden_1 = tf.layers.dense(x,
			 128,
			 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
			 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
			 activation=tf.nn.relu,
			 name='hidden_layer')
		hidden_2 = tf.layers.dense(
			 hidden_1,
			 128,
			 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
			 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
			 activation=tf.nn.relu,
			 name='hidden_layer_2')
		output = tf.layers.dense(
			 hidden_2,
			 10,
			 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
			 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0),
			 name='output')
		return output