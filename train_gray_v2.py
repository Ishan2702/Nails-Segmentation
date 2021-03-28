from __future__ import print_function
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import numpy as np
import glob
import tensorflow as tf
#import "tensorflow/python/ops/nn_ops.py"
import math
import cv2

directory = "./"
directory_data = "./gf_data/"
directory_log = "./logs"
directory_log_segmentation = "./logs/segmentation/"

if not os.path.exists(directory_log):
    os.mkdir(directory_log)

if not os.path.exists(directory_log_segmentation):
    os.mkdir(directory_log_segmentation)

if not os.path.exists(directory_data):
    os.mkdir(directory_data)

size_im = 200,200
n_input = 200*200
n_output = n_input
dim = n_output


learning_rate=0.0001

dimension = 200


#directory_test_im = "/media/avinash/F866FDA466FD6432/gromeefy/sem_seg_nails/nail_test/"
#directory_test_lable = "/media/avinash/F866FDA466FD6432/gromeefy/sem_seg_nails/nail_test_output/"
#im_test = Image.open(directory_test_im + "*.png")	#this opens the image file in image format not in array format
#im_test_lable = Image.open(directory_test_lable + "*.png")	#this opens the image file in image format not in array format

trainim = np.load(directory + "images.npy")
lableim = np.load(directory + "lables.npy")
#trainim = np.reshape(trainim, (-1,200,200,1))
train_image_display = trainim[40]
#cv2.imshow('train_image_display',train_image_display)
#cv2.waitKey(0)
#lableim = np.reshape(lableim, (-1,200,200,1))
lable_image_display = lableim[40]
#cv2.imshow('lable_image_display',lable_image_display)
#cv2.waitKey(0)
#print('Y_data shape:', np.array(lableim).shape)
#print('x_data shape:', np.array(trainim).shape)
#testimgs = np.array(im_test.resize(size_im))
#testlabels = np.array(im_test_lable.resize(size_im))

totsize = 287

trainim = trainim/float(255)
#train_image_display = (trainim[40]*255)
#cv2.imshow('train_image_display',train_image_display)
#cv2.waitKey(0)
lableim = lableim/float(255)
#testimgs = testimgs/255
#testlabels = testlabels/255

hin, win = 200,200	#image size

batch_size = 41 #287/7 = 41(integer value)

epochs = 100000
display_step = 5

activation_fn = tf.nn.relu

def saveWeights(le1, le2, le3, le4, le5, ld5, ld4, ld3, ld2, ld1):
	np.save(directory_data + "/le1.npy", le1)
	np.save(directory_data + "/le2.npy", le2)
	np.save(directory_data + "/le3.npy", le3)
	np.save(directory_data + "/le4.npy", le4)
	np.save(directory_data + "/le5.npy", le5)
	np.save(directory_data + "/ld5.npy", ld5)
	np.save(directory_data + "/ld4.npy", ld4)
	np.save(directory_data + "/ld3.npy", ld3)
	np.save(directory_data + "/ld2.npy", ld2)
	np.save(directory_data + "/ld1.npy", ld1)

def saveBiases(be1, be2, be3, be4, be5, bd5, bd4, bd3, bd2, bd1):
	np.save(directory_data + "/be1.npy", be1)
	np.save(directory_data + "/be2.npy", be2)
	np.save(directory_data + "/be3.npy", be3)
	np.save(directory_data + "/be4.npy", be4)
	np.save(directory_data + "/be5.npy", be5)
	np.save(directory_data + "/bd5.npy", bd5)
	np.save(directory_data + "/bd4.npy", bd4)
	np.save(directory_data + "/bd3.npy", bd3)
	np.save(directory_data + "/bd2.npy", bd2)
	np.save(directory_data + "/bd1.npy", bd1)

x = tf.placeholder(tf.float32, shape = (batch_size,dimension,dimension,1), name = 'inputs')
y = tf.placeholder(tf.float32, shape = (batch_size,dimension,dimension,1), name = 'targets')
	
#WEIGHTS AND BIASES
n1 = 16
n2 = 32
n3 = 64
ksize = 5

ll = 0
ul = 0


def nextbatch(batch_i):
	global ll
	global hl
	
	ll = batch_i*batch_size
	ul = batch_i*batch_size + (batch_size)
	#print (ll)
	#print (ul)
	
	tempy = lableim[ll:ul].copy()
	#print('tempy_data shape:', np.array(tempy).shape)
	tempx = trainim[ll:ul].copy()
	#print('tempx_data shape:', np.array(tempx).shape)
	#image_image_display = np.reshape(tempx[40],(200,200))
	#image_image_display = image_image_display*255
	#plt.imshow(image_image_display)
	#plt.show()
	#cv2.waitKey(0)
	lable_image_display = tempy[40]*255
	#cv2.imshow('batchlable',lable_image_display)
	#cv2.waitKey(0)




	#print tempnoisy.shape
	#tempx = tempx.reshape(batch_size, 200*200)
	#tempy = tempy.reshape(batch_size, 200*200)
	#print(tempy)
	#print(tempx)
	#ll = ll+incr
	return tempy, tempx

#randomly initialized the weights
weights = {
	'ce1' : tf.Variable(tf.random_normal([ksize, ksize, 1, n1], stddev = 0.1)),
	'ce2' : tf.Variable(tf.random_normal([ksize, ksize, n1, n2], stddev = 0.1)),
	'ce3' : tf.Variable(tf.random_normal([ksize, ksize, n2, n3], stddev = 0.1)),
        'ce4' : tf.Variable(tf.random_normal([ksize, ksize, n3, n4], stddev = 0.1)),
        'ce5' : tf.Variable(tf.random_normal([ksize, ksize, n4, n5], stddev = 0.1)),
        'cd5' : tf.Variable(tf.random_normal([ksize, ksize, n5, n4], stddev = 0.1)),
        'cd4' : tf.Variable(tf.random_normal([ksize, ksize, n4, n3], stddev = 0.1)),
	'cd3' : tf.Variable(tf.random_normal([ksize, ksize, n3, n2], stddev = 0.1)),
	'cd2' : tf.Variable(tf.random_normal([ksize, ksize, n2, n1], stddev = 0.1)),
	'cd1' : tf.Variable(tf.random_normal([ksize, ksize, 1, n1], stddev = 0.1))
}

biases = {
	'be1' : tf.Variable(tf.random_normal([n1], stddev = 0.1)),
	'be2' : tf.Variable(tf.random_normal([n2], stddev = 0.1)),
	'be3' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
        'be4' : tf.Variable(tf.random_normal([n4], stddev = 0.1)),
        'be5' : tf.Variable(tf.random_normal([n5], stddev = 0.1)),
        'bd5' : tf.Variable(tf.random_normal([n4], stddev = 0.1)),
        'bd4' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bd3' : tf.Variable(tf.random_normal([n2], stddev = 0.1)),
	'bd2' : tf.Variable(tf.random_normal([n1], stddev = 0.1)),
	'bd1' : tf.Variable(tf.random_normal([1], stddev = 0.1))
}

# convolution autoencoder layer definition
def cae (_X, _W, _b) :#, _keepprob):
	#_input_r = tf.reshape(_X, shape=[batch_size,dimension,dimension,1])
	#print(_X.shape)
	#image_image_display = _X[40]
	#print(image_image_display)
	#image_image_display = np.reshape(_X[40],(200,200))
	#image_image_display = image_image_display*255
	#plt.imshow(image_image_display)
	#plt.show()



	#print ('input_r =',_input_r.get_shape())

	#ENCODER
	_ce1 = tf.nn.relu(tf.add(tf.nn.conv2d(_X, _W['ce1'], strides = [1,1,1,1], padding='SAME'), _b['be1']))
	#_ce1 = tf.nn.dropout(_ce1, _keepprob)
	_ce2 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce1, _W['ce2'], strides = [1,1,1,1], padding='SAME'), _b['be2']))
	#_ce2 = tf.nn.dropout(_ce2, _keepprob)
	_ce3 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce2, _W['ce3'], strides = [1,1,1,1], padding='SAME'), _b['be3']))
	#_ce3 = tf.nn.dropout(_ce3, _keepprob)
	
	_ce4 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce2, _W['ce4'], strides = [1,1,1,1], padding='SAME'), _b['be4']))
	
	_ce5 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce2, _W['ce5'], strides = [1,1,1,1], padding='SAME'), _b['be5']))
	
	#print ('ce3 =', _ce3.get_shape())
	#DECODER

	_cd5 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_ce5, _W['cd5'], tf.stack([tf.shape(_X)[0],dimension,dimension,n2]), strides = [1,1,1,1], padding = 'SAME'), _b['bd5']))

	_cd4 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_ce4, _W['cd4'], tf.stack([tf.shape(_X)[0],dimension,dimension,n2]), strides = [1,1,1,1], padding = 'SAME'), _b['bd4']))
	
	_cd3 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_ce3, _W['cd3'], tf.stack([tf.shape(_X)[0],dimension,dimension,n2]), strides = [1,1,1,1], padding = 'SAME'), _b['bd3']))
	#_cd3 = tf.nn.dropout(_cd3, _keepprob)	
	_cd2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_cd3, _W['cd2'], tf.stack([tf.shape(_X)[0],dimension,dimension,n1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd2']))
	#_cd2 = tf.nn.dropout(_cd2, _keepprob)	
	_cd1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_cd2, _W['cd1'], tf.stack([tf.shape(_X)[0],dimension,dimension,1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd1']))
	#_cd1 = tf.nn.dropout(_cd1, _keepprob)	
	_out = _cd1
	return _out

print ("Network ready")

#keepprob = tf.placeholder(tf.float32)
#pred = cae(x, weights, biases, keepprob)
pred = cae(x, weights, biases)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = pred) #tf.reshape(x, shape=[-1,dimension,dimension,1])
cost = tf.reduce_mean(loss)

optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

print("All functions ready")

sess = tf.Session()
sess.run(init)
#mean_img = np.mean(mnist.train.images, axis = 0)	
#mean_img = np.zeros((784))
#Fit all the training data
 
saver = tf.train.Saver()

print("Start training")
for epoch_i in range(epochs):
	num_batch = int(totsize/batch_size)
	#print(num_batch)
	for batch_i in range(num_batch):
		#print(batch_i)
		batch_y, batch_x = nextbatch(batch_i)
		#print ('batch_x =', np.array(batch_x).shape)
		#print ('batch_y =', np.array(batch_y).shape)
		#sess.run(optm, feed_dict = {x:batch_x, y:batch_y, keepprob:1})
		sess.run(optm, feed_dict = {x:batch_x, y:batch_y}) #batch_x is image and batch_y is nails
	#ll=0
	#ul=0
	#print("[%02d/%02d] cost: %.4f" % (epoch_i, epochs, sess.run(cost, feed_dict={x : batch_x, y: batch_y, keepprob :1})))
	print("[%02d/%02d] cost: %.4f" % (epoch_i, epochs, sess.run(cost, feed_dict={x : batch_x, y: batch_y})))
	if epoch_i % display_step == 0 or epoch_i == epochs - 1:
		saver.save(sess, "logs/segmentation/latestmodel.ckpt")
		le1 = sess.run(weights['ce1'])
		le2 = sess.run(weights['ce2'])
		le3 = sess.run(weights['ce3'])
		le4 = sess.run(weights['ce4'])
		le5 = sess.run(weights['ce5'])
		ld5 = sess.run(weights['cd5'])
		ld4 = sess.run(weights['cd4'])
		ld3 = sess.run(weights['cd3'])
		ld2 = sess.run(weights['cd2'])
		ld1 = sess.run(weights['cd1'])
		be1 = sess.run(biases['be1'])
		be2 = sess.run(biases['be2'])
		be3 = sess.run(biases['be3'])
		be4 = sess.run(biases['be4'])
		be5 = sess.run(biases['be5'])
		bd5 = sess.run(biases['bd5'])
		bd4 = sess.run(biases['bd4'])
		bd3 = sess.run(biases['bd3'])
		bd2 = sess.run(biases['bd2'])
		bd1 = sess.run(biases['bd1'])
		saveWeights(le1,le2,le3,le4,le5,ld5,ld4,ld3,ld2,ld1)
		saveBiases(be1,be2,be3,be4,be5,bd5,bd4,bd3,bd2,bd1)
