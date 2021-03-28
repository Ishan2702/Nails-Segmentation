from __future__ import print_function
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import math
import cv2

directory = "./"
directory_result = "./result/"
directory_input = "./test_im/"

size_im = 200,200
n_input = 200*200
n_output = n_input
dim = n_output


learning_rate=0.0001

dimension = 200

im_array = []


#directory_test_im = "/media/avinash/F866FDA466FD6432/gromeefy/sem_seg_nails/nail_test/"
#directory_test_lable = "/media/avinash/F866FDA466FD6432/gromeefy/sem_seg_nails/nail_test_output/"
#im_test = Image.open(directory_test_im + "*.png")	#this opens the image file in image format not in array format
#im_test_lable = Image.open(directory_test_lable + "*.png")	#this opens the image file in image format not in array format

#no_of_im
no_of_im = 0
files = glob.glob (directory_input+"*.png")
for myFile in sorted (files):
	img = cv2.imread(myFile)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (200, 200))
	im_array.append (img)
	no_of_im = no_of_im + 1
	#return no_of_im
im_array = np.array(im_array)
#o_of_im = im_array.shape(0)

def save_images(samples):
    #fig = plt.figure(figsize=(28, 28))
    #gs = gridspec.GridSpec(8, 8)
    #gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        #sample = sample.reshape(2,28)
        #print(sample)
        cv2.imwrite(directory_result + "/%s.png" % str(i).zfill(3), sample*255, [cv2.IMWRITE_PNG_COMPRESSION,0]) #"""np.multiply(sample,255)"""
        #cv2.imwrite(sample_dir + "/%s.png" % str(i).zfill(3), sample*255)
#trainim = np.load(directory + "im_array.npy")
#lableim = np.load(directory + "seg_im_array.npy")
#print('Y_data shape:', np.array(lableim).shape)
#print('x_data shape:', np.array(trainim).shape)
#testimgs = np.array(im_test.resize(size_im))
#testlabels = np.array(im_test_lable.resize(size_im))

totsize = 287

#

im_array = im_array/float(255)
plt.imshow(im_array[0]*255)
plt.show()
#cv2.waitKey(0)
im_array = np.reshape(im_array,(-1,200,200,1))
#trainim = trainim/255
#lableim = lableim/255
#testimgs = testimgs/255
#testlabels = testlabels/255

hin, win = 200,200	#image size


#batch_size = 41 #281/7 = 41(integer value)

epochs = 10000
display_step = 5

activation_fn = tf.nn.relu

dle1 = np.load("model/le1.npy")
dle2 = np.load("model/le2.npy")
dle3 = np.load("model/le3.npy")
dld3 = np.load("model/ld3.npy")
dld2 = np.load("model/ld2.npy")
dld1 = np.load("model/ld1.npy")

dbe1 = np.load("model/be1.npy")
dbe2 = np.load("model/be2.npy")
dbe3 = np.load("model/be3.npy")
dbd3 = np.load("model/bd3.npy")
dbd2 = np.load("model/bd2.npy")
dbd1 = np.load("model/bd1.npy")


x = tf.placeholder(tf.float32, shape = (no_of_im,dimension,dimension,1), name = 'inputs')
#y = tf.placeholder(tf.float32, name = 'targets')
	
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
	tempx = trainim[ll:ul].copy()
	#print('tempx_data shape:', np.array(tempx).shape)
	tempy = lableim[ll:ul].copy()
	#print tempnoisy.shape
	#tempx = tempx.reshape(batch_size, n_input)
	#tempy = tempy.reshape(batch_size, n_input)
	#print(tempy)
	#print(tempx)
	#ll = ll+incr
	return tempy, tempx

#randomly initialized the weights
weights = {
	'ce1' : tf.Variable(dle1),
	'ce2' : tf.Variable(dle2),
	'ce3' : tf.Variable(dle3),
	'cd3' : tf.Variable(dld3),
	'cd2' : tf.Variable(dld2),
	'cd1' : tf.Variable(dld1)
}

biases = {
	'be1' : tf.Variable(dbe1),
	'be2' : tf.Variable(dbe2),
	'be3' : tf.Variable(dbe3),
	'bd3' : tf.Variable(dbd3),
	'bd2' : tf.Variable(dbd2),
	'bd1' : tf.Variable(dbd1)
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
	#print ('ce3 =', _ce3.get_shape())
	#DECODER
	_cd3 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_ce3, _W['cd3'], tf.stack([tf.shape(_X)[0],dimension,dimension,n2]), strides = [1,1,1,1], padding = 'SAME'), _b['bd3']))
	#_cd3 = tf.nn.dropout(_cd3, _keepprob)	
	_cd2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_cd3, _W['cd2'], tf.stack([tf.shape(_X)[0],dimension,dimension,n1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd2']))
	#_cd2 = tf.nn.dropout(_cd2, _keepprob)	
	_cd1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(_cd2, _W['cd1'], tf.stack([tf.shape(_X)[0],dimension,dimension,1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd1']))
	#_cd1 = tf.nn.dropout(_cd1, _keepprob)	
	_out = _cd1
	return _out

print ("Network ready")

keepprob = tf.placeholder(tf.float32)

pred = cae(x, weights, biases)

init = tf.global_variables_initializer()

print("All functions ready")

sess = tf.Session()
sess.run(init)
#mean_img = np.mean(mnist.train.images, axis = 0)	
#mean_img = np.zeros((784))
#Fit all the training data
 
#saver = tf.train.Saver()
#no_of_im = read_im()
print(no_of_im)
print('im_array shape:', np.array(im_array).shape)

print("Start testing")
samples = sess.run(pred, feed_dict = {x:im_array})
cae_in = np.reshape(im_array[0],(200,200))
plt.imshow(cae_in)
plt.show()
sam_test = np.reshape(samples[0],(200,200))
plt.imshow(sam_test*255)
plt.show()

save_images(samples)

"""
for epoch_i in range(epochs):
	num_batch = int(totsize/batch_size)
	#print(num_batch)
	for batch_i in range(num_batch):
		#print(batch_i)
		batch_y, batch_x = nextbatch(batch_i)
		#print ('batch_x =', np.array(batch_x).shape)
		#print ('batch_y =', np.array(batch_y).shape)
		sess.run(optm, feed_dict = {x:batch_x, y:batch_y, keepprob:1})
	#ll=0
	#ul=0
	print("[%02d/%02d] cost: %.4f" % (epoch_i, epochs, sess.run(cost, feed_dict={x : batch_x, y: batch_y, keepprob :1})))
	if epoch_i % display_step == 0 or epoch_i == epochs - 1:
		saver.save(sess, "logs/segmentation/latestmodel.ckpt")
		le1 = sess.run(weights['ce1'])
		le2 = sess.run(weights['ce2'])
		le3 = sess.run(weights['ce3'])
		ld3 = sess.run(weights['cd3'])
		ld2 = sess.run(weights['cd2'])
		ld1 = sess.run(weights['cd1'])
		be1 = sess.run(biases['be1'])
		be2 = sess.run(biases['be2'])
		be3 = sess.run(biases['be3'])
		bd3 = sess.run(biases['bd3'])
		bd2 = sess.run(biases['bd2'])
		bd1 = sess.run(biases['bd1'])
		saveWeights(le1,le2,le3,ld3,ld2,ld1)
		saveBiases(be1,be2,be3,bd3,bd2,bd1)
"""