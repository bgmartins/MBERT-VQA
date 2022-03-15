import os
import glob
import cv2
from PIL import Image
from torchvision import transforms
import h5py
from tqdm import tqdm
import numpy as np




LR = {'data_dir': 'Remote_Sensing_Datasets/RSVQA_LR',
           'image_dir': 'Images',
           'resize_dir': 'Images_Resized',
           'extension': 'jpg'}

HR = {'data_dir': 'Remote_Sensing_Datasets/RSVQA_HR',
           'image_dir': 'Images',
           'resize_dir': 'Images_Resized',
           'extension': 'jpg'}

BEN = {'data_dir': 'Remote_Sensing_Datasets/RSVQA_BEN',
           'image_dir': 'Images',
           'resize_dir': 'Images_Resized',
           'extension': 'tif'}

data_info = BEN # CHOOSE HERE

print('Data info', data_info)
data_dir = os.path.join('..', data_info['data_dir'])

tfm = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224)])

impaths = []
for aux in glob.glob(os.path.join(data_dir, data_info['image_dir']) + f'/*.{data_info["extension"]}'):
    impaths.append(aux)

print('len images',len(impaths))

SAVE_DIR = os.path.join(data_dir, data_info['resize_dir'])
i=0
for i, path in enumerate(tqdm(impaths)):

    # Read images
    img = Image.open(impaths[i]).convert("RGB")

    if i < 3:
        print('orig shape', img.size)
    img = tfm(img)
    if i < 3:
        print('tfm shape', img.size)

    p, ext = os.path.splitext(impaths[i])
    file = p.split('/')[-1]
    img.save(os.path.join(SAVE_DIR, file + ".jpg"))
    
    i+=1






#attempt to build hdf5 file
# with h5py.File(os.path.join(data_dir, 'RSVQA_LR.hdf5'),'a') as h:

#     # Create dataset inside HDF5 file to store images
#     images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

#     print("\nReading %s images and labels, storing to file...\n")

#     # encode images and labels
#     for i, path in enumerate(tqdm(impaths)):

#         # Read images
#         #img = Image.open(impaths[i]).convert("RGB")

#         img = cv2.imread(impaths[i])
#         print('imgread', img.shape)
#         w=224;h=224
#         (h_orig, w_orig) = img.shape[:2]
#         if h_orig < w_orig:
#             img=image_resize(img, height = h)
#         else:
#             img=image_resize(img, width= w)
#         center = (img.shape[0]/2,img.shape[1]/2)
#         x = center[1] - w/2
#         y = center[0] - h/2
#         img = img[int(y):int(y+h), int(x):int(x+w)]
#         print('final img shape',img.shape)
#         img = img.transpose(2, 0, 1)

#         assert img.shape == (3, 224, 224)
#         assert np.max(img) <= 255

#         # Save image to HDF5 file
#         images[i] = img

#image resize function from https://stackoverflow.com/a/44659589
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized