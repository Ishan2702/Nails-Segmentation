import os, sys
import numpy as np
import glob
import cv2

directory_input_nails = "C:/Users/ishan/Desktop/nails images/Augmented Nails/"
directory_input_lables = "C:/Users/ishan/Desktop/nails images/Segmented Nails1/"
directory_output = "C:/Users/ishan/Downloads/Edited_Model/tools/numpy/"

if not os.path.exists(directory_output):
	os.mkdir(directory_output)

im_array = []
lable_array = []

files = glob.glob (directory_input_nails+"*.jpg")
for myFile in files:
	img = cv2.imread(myFile)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_CUBIC)
	im_array.append (img)
print('nails shape:', np.array(im_array).shape)
files = glob.glob (directory_input_lables+"*.png")
for myFile in files:
	img = cv2.imread(myFile)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_CUBIC)
	lable_array.append (img)
print('lable shape:', np.array(lable_array).shape)


np.save(directory_output + 'nails.npy', im_array)
np.save(directory_output + 'labels.npy', lable_array)
