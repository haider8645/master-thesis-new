from PIL import Image
import glob
from random import shuffle

def shuffle_images(addrs):
    shuffle(addrs)
    print 'Length of shuffled addresses'
    print len(addrs)
    return addrs

def save_images(save_location,addresses):
    for i in range (len(addresses)):
        img = Image.open(addresses[i])
        img.save(save_location +'shredded' +str(i) + '.png', 'PNG')
    return

trash_pics = '/home/haider/Desktop/KIPro Data/Data/data_transformed/validation/spaene/*.png'
addrs = glob.glob(trash_pics)
print 'Length of addrs'
print len(addrs)

shuffled_addresses = shuffle_images(addrs)

save_images('/home/haider/Desktop/KI-pro-data-shuffled/',shuffled_addresses)

print '\nFinished processing & shuffling all images'
