from PIL import Image
import glob, os

counter=1
files = glob.glob('D:/nails images/Dataset/*.jpg')

for filename in files:
    print (filename)
    
    image = Image.open(filename)
    image=image.resize((512,512), Image.ANTIALIAS)
    paddedfilename = 'D:/nails images/data512_512/inputs/'+ str(counter).zfill(4) + '.jpg'
    image.save(paddedfilename)

    maskfilename=filename.replace(".jpg","nails.png")
    image = Image.open(maskfilename)
    image=image.convert('RGBA')

    pixels = image.load()

    for i in range(image.size[0]): # for every pixel:
        for j in range(image.size[1]):
            if pixels[i,j][3] == 0:
                pixels[i,j] = (0,0,0)
            else:
                pixels[i,j] = (255,255,255)
    image = image.convert('RGB')
    paddedfilename = 'D:/nails images/data512_512/targets/'+ str(counter).zfill(4) + '.jpg'
    image=image.resize((512,512), Image.ANTIALIAS)
    image.save(paddedfilename)
    counter = counter + 1
##    image.show()

