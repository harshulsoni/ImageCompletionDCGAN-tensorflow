import tensorflow as tf
import numpy as np
import parameters as parameters

print(tf.__version__)

def DCGAN(batch_size=parameters.batch_size, learning_rate=1e-4):
	with tf.variable_scope('generator'):
		#z=tf.random_normal([batch_size, 64, 64, 3], mean=0, stddev=0.01, dtype=tf.float32)	
		z=tf.placeholder(tf.float32, [batch_size, parameters.z_len])
		training=tf.placeholder(tf.bool, name='training_phase')
		GenTrain=tf.placeholder(tf.bool, name='Generator_Train_on')
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
			outputs=tf.reshape(layer3, [batch_size, 4, 4, 1024])
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
		x_images=tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
		with tf.variable_scope('prepare_input'):
			y_i=tf.ones([batch_size, 1], tf.float32)
			y_f=tf.zeros([batch_size, 1], tf.float32)
			y_images=tf.concat([y_f, y_i], 1)
			y_fake=tf.concat([y_i, y_f], 1)
			def callablefnc(input):
				return input
			x=tf.cond(GenTrain, lambda: callablefnc(dconv4),lambda: tf.concat([dconv4, x_images], 0))
			x=tf.reshape(x, [-1,64,64,3])
			y=tf.cond(GenTrain, lambda: callablefnc(y_images),lambda: tf.concat([y_fake, y_images], 0))
			y=tf.reshape(y, [-1, 2])
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
			q=int(conv4_shape[1])*int(conv4_shape[2])*int(conv4_shape[3])
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
		with tf.variable_scope('loss'):
			loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer3, labels=y))
		with tf.variable_scope('training'):
			d_train=tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=d_variables)
			g_train=tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=g_variables)
		saver=tf.train.Saver()

	with tf.variable_scope('ImageCompletion'):
		mask=tf.placeholder(tf.float32, [None, 64, 64, 3])
		#x_images
		#z
		#set GenTrain =True
		contextual_loss=tf.reduce_sum(tf.reduce_sum(tf.abs(tf.multiply(mask, dconv4)-tf.multiply(mask, x_images)), 1), 0)
		perceptual_loss=loss
		total_loss=contextual_loss+0.1*perceptual_loss
		grad_loss=tf.gradients(total_loss, z)
	
	return dict(
		x=x_images,
		mask=mask,
		z=z,
		training=training,
		GenTrain=GenTrain,
		loss=loss,
		d_train=d_train,
		g_train=g_train,
		output=dconv4,
		saver=saver
	)

dc=DCGAN()

