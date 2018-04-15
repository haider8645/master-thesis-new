from PIL import Image
import glob
import os

trash_pics = '/home/haider/scripts-test-area/data/*.jpg' #reading the trash data set files

addrs = glob.glob(trash_pics)
print len(addrs)
x=0
for images in range(0,len(addrs)):

#############################################
#### CODE TO CROP EACH IMAGE INTO 12 SECTIONS
    img=Image.open(addrs[images])
    startwidth=107
    startheight=0
    width = 491
    height = 384
    head, tail = os.path.split(addrs[images])
    print tail
    img_crop = img.crop((startwidth, startheight, width, height))
    img_crop.save('/home/haider/caffe/python-scripts/trashnet/data-center-384-labels/'+str(tail), 'JPEG') 
    x=x+1
#############################################
