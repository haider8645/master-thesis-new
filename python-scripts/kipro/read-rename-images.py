from PIL import Image
import os
import glob
from random import shuffle
import numpy as np



def shuffle_images(addrs):
    shuffle(addrs)
    print 'Length of shuffled addresses'
    print len(addrs)
    return addrs

def save_images(save_location,addresses):
    for i in range (len(addresses)):
        img = Image.open(addresses[i])
        img.save(save_location +'cardboard2' +str(i), 'PNG')

    return


#Size of images
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384

trash_pics = '/home/haider/Desktop/KIPro Data/Data/data_original/data/data_original/training/b12/*.png'
addrs = glob.glob(trash_pics)
print 'Length of addrs' 
print len(addrs)

shuffled_addresses = shuffle_images(addrs)

save_images('/home/haider/Desktop/KI-pro-data-shuffled/',shuffled_addresses)

print '\nFinished processing & shuffling all images'
