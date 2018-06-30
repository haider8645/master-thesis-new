import Image
import glob
import os
import random

trash_pics = '/home/lod/datasets/trashnet/data/dataset-resized/glass/*.jpg' #reading the trash data set files

addrs = glob.glob(trash_pics)
print len(addrs)
random.shuffle(addrs)

x=0
for images in range(len(addrs)):
    
#############################################
#### CODE TO CROP EACH IMAGE INTO SECTIONS
#    size=(512,384)
    print addrs[images]
    img=Image.open(addrs[images])
    head, tail = os.path.split(addrs[images])
    width = 277
    height = 277
    startwidth=50
    startheight=50
#    for j in range(0,2):
#        for i in range(0,4):
    img_crop = img.crop((startwidth, startheight, width, height))
           # if (j == 1 and i == 2): #save only a selected images
    img_crop.save('/home/lod/datasets/trashnet/data/dataset-resized/resized_to_227/'+str(tail), 'JPEG')
 #           print img_crop.size
#            startwidth=startwidth+384
#            width=width+384 
#            x = x+1
#        startheight=startheight+384
#        height = height + 384
#        startwidth = 0
#        width = 384     
#############################################
