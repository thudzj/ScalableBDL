import numpy as np
from random import random, choice
from scipy.ndimage.filters import gaussian_filter
import cv2
from io import BytesIO
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GeneratedDataAugment(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        img = np.array(img)

        if random() < self.args.blur_prob:
            sig = sample_continuous(self.args.blur_sig)
            gaussian_blur(img, sig)

        if random() < self.args.jpg_prob:
            method = sample_discrete(self.args.jpg_method)
            qual = sample_discrete(list(range(self.args.jpg_qual[0], self.args.jpg_qual[1] + 1)))
            img = jpeg_from_key(img, qual, method)
        return Image.fromarray(img)

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

def jpeg_from_key(img, compress_val, key):
    if key == 'cv2':
        return cv2_jpg(img, compress_val)
    else:
        return pil_jpg(img, compress_val)
