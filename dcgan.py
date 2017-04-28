import tensorflow as tf
import numpy as np
import parameters as parameters

print(tf.__version__)

def DCGAN():
	with tf.variable_scope('generator'):
		z=tf.placeholder(tf.float32, [parameters.batch_size, parameters.z_len])
		training=tf.placeholder(tf.bool, name='training_phase')
		def fully_connected_layer(input, name, in_size, out_size):
			with tf.variable_scope(name):
				weights=tf.get_variable("weights", [in_size, out_size], initializer=tf.random_normal_initializer(0, 0.01))
				#biases=tf.get_variable("biases", [out_size], initializer=tf.random_normal_initialier(0, 0.01))
				#outputs=tf.add(tf.matmul(input, weights), biases)
				outputs=tf.matmul(input, weights)
			return outputs
		with tf.variable_scope('Fully_Connected'):
			layer1=fully_connected_layer(z, 'layer1', parameters.z_len, 256)
			layer2=fully_connected_layer(layer1, 'layer2', 256, 1024)
			layer3=fully_connected_layer(layer2, 'layer3', 1024, 1024*4*4)
		with tf.variable_scope('flattening'):
			outputs=tf.reshape(layer3, [-1, 4, 4, 1024])
			outputs=tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
		def deconvlayer(input, name, in_channels, out_channels):
			with tf.variable_scope(name):
				#conv=tf.nn.conv2d_transpose(input, w_filter, strides=[1, 2, 2, 1], padding='SAME', name='conv')
				conv=tf.layers.conv2d_transpose(input, out_channels, [5, 5], strides=(2, 2), padding='SAME')
				outputs = tf.nn.relu(tf.layers.batch_normalization(conv, training=training), name='outputs')
			return outputs
		dconv1=deconvlayer(outputs, 'Deconvolution1', 1024, 512)
		dconv2=deconvlayer(dconv1, 'Deconvolution2', 512, 256)
		dconv3=deconvlayer(dconv2, 'Deconvolution3', 256, 128)
		dconv4=deconvlayer(dconv3, 'Deconvolution4', 128, 3)
		g_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
	'''return dict{
		output=dconv4
		variables=g_variables
	}'''

	with tf.variable_scope('discriminator'):
		x_images=tf.placeholder(tf.float32, [parameters.batch_size, 64, 64, 3])
		training=tf.placeholder(tf.bool, name='training_phase')
		#y_images=tf.placeholder(tf.float32, [parameters.batch_size, parameters.num_classes])
		with tf.variable_scope('prepare_input_and_shuffle'):
			y_i=tf.ones([parameters.batch_size, 1], tf.float32)
			y_f=tf.zeros([parameters.batch_size, 1], tf.float32)
			y_images=tf.concat([y_f, y_i], 1)
			y_fake=tf.concat([y_i, y_f], 1)
			x=tf.concat([dconv4, x_images], 0)
			x_shape=x.get_shape()
			x=tf.reshape(x, [2*parameters.batch_size, -1])
			y=tf.concat([y_fake, y_images], 0)
			y=tf.reshape(y, [2*parameters.batch_size, -1])
			x_temp=tf.concat([x, y], 1)
			x_temp=tf.random_shuffle(x_temp, name='shuffle')
			x=tf.slice(x_temp, [0, 0], [2*parameters.batch_size, int(x_shape[1])*int(x_shape[2])*int(x_shape[3])], name='slicing_x')
			x=tf.reshape(x, [2*parameters.batch_size, int(x_shape[1]),int(x_shape[2]),int(x_shape[3])])
			y=tf.slice(x_temp, [0, int(x_shape[1])*int(x_shape[2])*int(x_shape[3])], [2*parameters.batch_size, 2], name='slicing_y')
		def convlayer(input, name, in_channels, out_channels):
			with tf.variable_scope(name):
				w_filter=tf.get_variable("w_filter", [5, 5, in_channels, out_channels], initializer=tf.random_normal_initializer(0, 0.01))
				#w_bias=tf.get_variable("w_filter", [out_channels], initializer=tf.random_normal_initialier(0, 0.01))
				#conv=tf.nn.conv2d(input, w_filter, strides=[1, 2, 2, 1], padding='SAME', name='conv')+w_bias
				conv=tf.nn.conv2d(input, w_filter, strides=[1, 2, 2, 1], padding='SAME', name='conv')
				outputs = tf.nn.relu(tf.layers.batch_normalization(conv, training=training), name='outputs')
			return outputs
		conv1=convlayer(x, 'Convolution1', 3, 64)
		conv2=convlayer(conv1, 'Convolution2', 64, 128)
		conv3=convlayer(conv2, 'Convolution3', 128, 256)
		conv4=convlayer(conv3, 'Convolution4', 256, 512)
		with tf.variable_scope('Conv4_Flattening'):
			conv4_shape=conv4.get_shape()
			q=conv4_shape[1]*conv4_shape[2]*conv4_shape[3]
			q=int(q)
			#h_pool2_flat=tf.reshape(h_pool2, [-1, 4*4*64])
			conv4_flat=tf.reshape(conv4, [-1, q])
		def fully_connected_layer(input, name, in_size, out_size):
			with tf.variable_scope(name):
				weights=tf.get_variable("weights", [in_size, out_size], initializer=tf.random_normal_initializer(0, 0.01))
				#biases=tf.get_variable("biases", [out_size], initializer=tf.random_normal_initialier(0, 0.01))
				#outputs=tf.add(tf.matmul(input, weights), biases)
				outputs=tf.matmul(input, weights)
			return outputs
		with tf.variable_scope('Fully_Connected'):
			layer1=fully_connected_layer(conv4_flat, 'layer1', q, 1024)
			layer2=fully_connected_layer(layer1, 'layer2', 1024, 256)
			layer3=fully_connected_layer(layer2, 'layer3', 256, 2)
		predictions=tf.nn.softmax(layer3)
		output=tf.argmax(predictions, 1)
		d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
		d_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer3, labels=y))
		d_train=tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=d_variables)
		saver=tf.train.Saver()
	'''return dict{
		x=x,
		y=y,
		training=training,
		variables=d_variables,
		loss=d_loss,
		train=d_train
		predictions=predictions,
		output=output,
		saver=saver
	}'''
