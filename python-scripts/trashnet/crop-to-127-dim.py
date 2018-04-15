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
#    size=(512,384)
    img=Image.open(addrs[images])
#    img.resize(size,Image.ANTIALIAS)
    width = 127
    height = 127
    startwidth=0
    startheight=0

    for j in range(0,3):
        for i in range(0,4):
            img_crop = img.crop((startwidth, startheight, width, height))
            if (i == 1 or i == 2): #save only a selected images, not all 8 images.
                img_crop.save('/home/haider/caffe/python-scripts/trashnet/data-center-cropped/output_image_' + str(x) + '.jpg', 'JPEG')
            startwidth=startwidth+128
            width=width+128
            x = x + 1 
        startheight=startheight+128
        height = height + 128
        startwidth = 0
        width = 127     
#############################################
