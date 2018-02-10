import tensorflow as tf
import numpy as np
import os
import util
import model

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where FMNIST is located')
#TODO: What to use as save dir?
flags.DEFINE_string('save_dir', 'hw1', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('proportion', 0.9, '')
flags.DEFINE_integer('max_epoch_num', 200, '')
FLAGS = flags.FLAGS



def main(argv):
	# load data
	train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
	train_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')
	# train_images, validation_images,

	# split into train and validate

	proportion = FLAGS.proportion
	train_images_2,  validation_image, train_labels_2, validation_labels = util.split_data(train_images,train_labels, proportion)
	#######################################

	#######################################
	validation_set_num_examples =  validation_image.shape[0]
	train_num_examples = train_images_2.shape[0]
	#test_num_examples = test_images.shape[0]



	# TODO: Rewrite in terms of objects and with new architecture
	# specify the network
	input_placeholder = tf.placeholder(tf.float32, [None, 784],
			name='input_placeholder1')
	output = model.layers(input_placeholder)
	# define classification loss
	y = tf.placeholder(tf.float32, [None, 10], name='label')

	#with tf.name_scope('cross_entropy') as scope:
	cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
	#cross_entropy = tf.reduce_mean(cross_entropy)

	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	REG_COEFF = 0.000001
<<<<<<< Updated upstream
	total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)
	# cross_entropy1 = tf.reduce_mean(total_loss)
=======
	total_loss1 = cross_entropy + REG_COEFF * sum(regularization_losses)
	total_loss = tf.reduce_mean(total_loss1)
>>>>>>> Stashed changes



	confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

	# set up training and saving functionality
	global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
	saver = tf.train.Saver()

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		# run training
		batch_size = FLAGS.batch_size
		lossControl = []
		for epoch in range(FLAGS.max_epoch_num):
			print('Epoch: ' + str(epoch))
			# run gradient steps and report mean loss on train data
			ce_vals = []

			for i in range(train_num_examples // batch_size):
				batch_xs = train_images_2[i*batch_size:(i+1)*batch_size, :]
				batch_ys = train_labels_2[i*batch_size:(i+1)*batch_size, :]
				batch_xs = batch_xs //255
				#_, train_ce = session.run([train_op, tf.reduce_mean(cross_entropy)], {x: batch_xs, y: batch_ys})
				_, train_ce = session.run([train_op, total_loss], {input_placeholder: batch_xs, y: batch_ys})
				ce_vals.append(train_ce)

			avg_train_ce = sum(ce_vals) / len(ce_vals)
		#   avg_test_cev = sum(ce_vals_v) / len(ce_vals_v)
		#   print('VALIDATION CROSS ENTROPY: ' + str(avg_test_cev))
		#   print('VALIDATION CONFUSION MATRIX:')
		#   print(str(sum(conf_mxs_v)))
			print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

			# report mean test loss
#            ce_vals = []
#            conf_mxs = []
#            for i in range(test_num_examples // batch_size):
#                batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
#                batch_ys = test_labels[i*batch_size:(i+1)*batch_size, :]
#                test_ce, conf_matrix = session.run([cross_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
#                ce_vals.append(test_ce)
#                conf_mxs.append(conf_matrix)
#            avg_test_ce = sum(ce_vals) / len(ce_vals)
#            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
#            #print('TEST CONFUSION MATRIX:')
#            #print(str(sum(conf_mxs)))

			ce_vals_v = []
			conf_mxs_v = []
			for i in range(validation_set_num_examples // batch_size):
				batch_xsv = validation_image[i*batch_size:(i+1)*batch_size, :]
				batch_ysv = validation_labels[i*batch_size:(i+1)*batch_size, :]
				batch_xsv = batch_xsv//255
<<<<<<< Updated upstream
				#batch_ysv = batch_ysv //255
=======
>>>>>>> Stashed changes
				test_cev, conf_matrix_v, _ = session.run([total_loss, confusion_matrix_op, output], {input_placeholder: batch_xsv, y: batch_ysv})
				ce_vals_v.append(test_cev)
				conf_mxs_v.append(conf_matrix_v)
			avg_test_cev = sum(ce_vals_v) / len(ce_vals_v)
			print('VALIDATION CROSS ENTROPY: ' + str(avg_test_cev))
			lossControl.append (avg_test_cev)
			if epoch > 30 :
				if (( np.average(lossControl)+1*np.std(lossControl)) < avg_test_cev):
					print('Early stopping happens at ' + str(epoch))
					print('the average+1std is: '+str(( np.average(lossControl)+np.std(lossControl))))
					break
			print('VALIDATION CONFUSION MATRIX:')
			print(str(sum(conf_mxs_v)))

		#print(output)
		path_prefix = saver.save(session, os.path.join(FLAGS.save_dir,"homework_1"), global_step=global_step_tensor)
		saver = tf.train.import_meta_graph(path_prefix + '.meta')
		saver.restore(session, path_prefix)
		return output
if __name__ == "__main__":
	tf.app.run()
