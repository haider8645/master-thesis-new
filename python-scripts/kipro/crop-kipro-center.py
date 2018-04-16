import Image
import glob
import os
import random

trash_pics = '/home/lod/master-thesis/LMDB-datasets/kipro/hendrik-dataset-transformed/test-train-images/KI-pro-data-shuffled/*.png' #reading the trash data set files

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
    width = 584
    height = 584
    startwidth=200
    startheight=200
#    for j in range(0,2):
#        for i in range(0,4):
    img_crop = img.crop((startwidth, startheight, width, height))
           # if (j == 1 and i == 2): #save only a selected images
    img_crop.save('/home/lod/master-thesis/LMDB-datasets/kipro/hendrik-dataset-transformed/center_cropped_hendrik_dataset/test_train_images/'+str(tail), 'PNG')
 #           print img_crop.size
#            startwidth=startwidth+384
#            width=width+384 
#            x = x+1
#        startheight=startheight+384
#        height = height + 384
#        startwidth = 0
#        width = 384     
#############################################
