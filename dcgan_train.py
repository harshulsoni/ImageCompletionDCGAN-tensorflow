from dcgan import *
import glob
import scipy.misc
from PIL import Image
import numpy as np

def get_image(name):	
	img=Image.open(name)
	#scipy.misc.imshow(img)
	#print (img.size)
	half_the_width = img.size[0] //2
	half_the_height = img.size[1] // 2
	img4 = img.crop(
	    (
	        half_the_width - 64,
	        half_the_height - 64,
	        half_the_width + 64,
	        half_the_height + 64
	    )
	)
	#scipy.misc.imshow(img4)
	#img4.save("img4.jpg")
	#scipy.misc.imshow(img4)
	k=scipy.misc.imresize(img4, [64, 64, 3], interp='bicubic')
	#scipy.misc.imshow(k)
	return k

def get_image_name_list():
	files = [file for file in glob.glob('data' + '/**/*.jpg', recursive=True)]
	return np.array(files)

def get_image_list(l):
	a=[]
	for i in l:
		a.append(get_image(i))
	return np.array(a)

images=get_image_name_list()
dcganModel=DCGAN()

sess=tf.Session()
writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
writer.close()

init=tf.global_variables_initializer()

sess.run(init)
#dcganModel['saver'].restore(sess, "trainedmodels/dcgan.model")
images_shape=images.shape
for i in range(parameters.num_epoch):
	print ("training epoch", i+1, "of", parameters.num_epoch)
	j=0
	g_loss=0
	d_loss=0
	#while j<images_shape and j+parameters.batch_size<images_shape:
		#x_train=images[j:j+parameters.batch_size]
		#j+=RNN_model.parameters.batch_size
	for ki in range(parameters.d_train_num):
		z=np.random.normal(0, 0.01, [parameters.batch_size, parameters.z_len])
		train_x=np.random.choice(images, parameters.batch_size)
		x_train=get_image_list(train_x)
		_, d_l=sess.run([dcganModel['d_train'], dcganModel['d_loss']], feed_dict={dcganModel['x']:x_train, dcganModel['z']:z, dcganModel['d_training']:True,dcganModel['g_training']:False})
		d_loss+=d_l
		print ('disc: ', ki+1, 'completed of', parameters.d_train_num)
	z=np.random.normal(0, 0.01, [parameters.batch_size, parameters.z_len])
	_, g_loss, g_images=sess.run([dcganModel['g_train'], dcganModel['g_loss'], dcganModel['g_output']], feed_dict={dcganModel['z']:z, dcganModel['g_training']:True,dcganModel['d_training']:False})
	#print (g_images[0][0])
	#print (g_images[5][0])
	scipy.misc.imsave('savedimages/'+'iteration'+str(i)+'_'+'0'+'.jpg',g_images[0])
	scipy.misc.imsave('savedimages/'+'iteration'+str(i)+'_'+'1'+'.jpg',g_images[5])
	scipy.misc.imsave('savedimages/'+'iteration'+str(i)+'_'+'2'+'.jpg',g_images[6])
	print ("Epoch: ", i+1, "completed. g_Loss: ", g_loss, ", d_loss: ", d_loss)
	if (i+1)%5==0:
		dcganModel['saver'].save(sess, "trainedmodels/dcgan.model")
myRnnModel['saver'].save(sess, "trainedmodels/dcgan.model")
sess.close()