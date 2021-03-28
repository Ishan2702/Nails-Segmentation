from glob import glob                                                           
import cv2

pngs = glob('D:/nails images/Augmented Nails1/*.jpg')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'png', img)
