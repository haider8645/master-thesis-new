from PIL import Image
import glob

output_pics = '/home/haider/caffe/python-scripts/trashnet/data/'
trash_pics = '/home/haider/scripts-test-area/data/*.jpg' #reading the trash data set files

addrs = glob.glob(trash_pics)
print len(addrs)
x=0
for images in range(0,len(addrs)):

#############################################
#### CODE TO CROP EACH IMAGE INTO 12 SECTIONS
    img=Image.open(addrs[images])
    width = 491
    height = 384
    startwidth=107
    startheight=0
    img_crop = img.crop((startwidth, startheight, width, height))
    img_crop.save('/home/haider/caffe/python-scripts/trashnet/data-center-256/output_image_' + str(x) + '.jpg', 'JPEG') 
    x=x+1
#############################################
