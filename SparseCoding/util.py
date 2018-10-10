# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          util
Author:             deepgray
--------------------------------------------
Description:
TODO:
似乎还是有问题（异常值太大了），需要一个简单的测试来证明算法的正确性


--------------------------------------------
Date:               18-7-2
Change Activity:

--------------------------------------------
"""
from skimage import io
from skimage import transform
from skimage.color import rgb2gray
from skimage.util import img_as_float
import logging


def load_image(path):
    try:
        image = io.imread(path)
    except IOError:
        logging.critical("File not found! Check whether file %s exists", path)
        return
    return image


def crop(image, patch_size):
    """
    crop image patches for train or test
    """
    im_list = list()
    i = int(image.shape[0] / patch_size[0])
    j = int(image.shape[1] / patch_size[1])

    height = i * patch_size[0]
    width = j * patch_size[1]
    new = transform.resize(image, (height, width))

    for x in range(i):
        for y in range(j):
            start_x = int(x * patch_size[0])
            start_y = int(x * patch_size[1])
            patch = new[start_x:start_x+patch_size[0], start_y:start_y+patch_size[1], :]
            # reshape to a column vector
            patch = rgb2gray(patch)
            patch = img_as_float(patch)
            # print(np.max(patch), np.min(patch))

            im_list.append(patch.reshape((patch_size[0]*patch_size[1])))
    return im_list, i, j
