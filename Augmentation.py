import cv2
from imgaug import augmenters as iaa
import numpy as np
import glob
from shutil import copyfile

counter=923
flipper = iaa.Fliplr(1.0) # always horizontally flip each input image
flipr=iaa.Flipud(1.0)
noise = iaa.Dropout(0.2)

files = glob.glob('C:/Users/ishan/Desktop/nails images/Dataset/*.jpg')

for filename in files:
    print (filename)
    maskfilename=filename.replace(".jpg","nails.png")
    
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    image1 = cv2.imread(maskfilename, cv2.IMREAD_UNCHANGED)
    
    paddedfilename = 'C:/Users/ishan/Desktop/nails images/Augmented Nails/'+ str(counter).zfill(6) + '.jpg'
    paddedfilename1 = 'C:/Users/ishan/Desktop/nails images/Segmented Nails/'+ str(counter).zfill(6) + 'nails.png'
    counter = counter + 1
    cv2.imwrite(paddedfilename,image)
    cv2.imwrite(paddedfilename1,image1)
    
    res = flipper.augment_image(image)
    res1= flipper.augment_image(image1)
    paddedfilename = 'C:/Users/ishan/Desktop/nails images/Augmented Nails/'+ str(counter).zfill(6) + '.jpg'
    paddedfilename1 = 'C:/Users/ishan/Desktop/nails images/Segmented Nails/'+ str(counter).zfill(6) + 'nails.png'
    counter = counter + 1
    cv2.imwrite(paddedfilename,res)
    cv2.imwrite(paddedfilename1,res1)

    res = flipr.augment_image(image)
    res1= flipr.augment_image(image1)
    paddedfilename = 'C:/Users/ishan/Desktop/nails images/Augmented Nails/'+ str(counter).zfill(6) + '.jpg'
    paddedfilename1 = 'C:/Users/ishan/Desktop/nails images/Segmented Nails/'+ str(counter).zfill(6) + 'nails.png'
    counter = counter + 1
    cv2.imwrite(paddedfilename,res)
    cv2.imwrite(paddedfilename1,res1)

    res = noise.augment_image(image)
##    res1 = flipper.augment_image(res1)
    paddedfilename = 'C:/Users/ishan/Desktop/nails images/Augmented Nails/'+ str(counter).zfill(6) + '.jpg'
    paddedfilename1 = 'C:/Users/ishan/Desktop/nails images/Segmented Nails/'+ str(counter).zfill(6) + 'nails.png'
    counter = counter + 1
    cv2.imwrite(paddedfilename,res)
    cv2.imwrite(paddedfilename1,image1)

cv2.waitKey(0)
