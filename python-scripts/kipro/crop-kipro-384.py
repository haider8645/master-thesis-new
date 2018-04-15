from PIL import Image
import glob
import os
import random

trash_pics = '/home/haider/Desktop/KI-pro-data-shuffled/*' #reading the trash data set files

addrs = glob.glob(trash_pics)
print len(addrs)
random.shuffle(addrs)

x=0
for images in range(len(addrs)):
    
#############################################
#### CODE TO CROP EACH IMAGE INTO 12 SECTIONS
#    size=(512,384)
    img=Image.open(addrs[images])
    head, tail = os.path.split(addrs[images])
    print addrs[images]
#    img.resize(size,Image.ANTIALIAS)
    width = 484
    height = 484
    startwidth=100
    startheight=100
    for j in range(0,2):
        for i in range(0,4):
            img_crop = img.crop((startwidth, startheight, width, height))
 #           if (i == 1 or i == 2): #save only a selected images, not all 8 images.
            img_crop.save('/home/haider/caffe/python-scripts/kipro/data-cropped-384/'+str(tail)+str(x), 'PNG')
 #           print img_crop.size
            startwidth=startwidth+384
            width=width+384 
            x = x+1
        startheight=startheight+384
        height = height + 384
        startwidth = 0
        width = 384     
#############################################
