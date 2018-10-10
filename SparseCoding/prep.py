# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          prep
Author:              deepgray
--------------------------------------------
Description:


--------------------------------------------
Date:               18-6-29
Change Activity:

--------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
from util import *
import math
import json
import time
import os


def train(train_path, model, save_path):
    """
    训练学习字典
    :param train_path:
    :param model:
    :param save_path:
    :return:
    """
    logging.info("Start Training model....")

    img_list = glob.glob(train_path + './*.jpg')
    # print("image list:", img_list)

    assert float.is_integer(math.sqrt(model.get_size()[1]))

    size = int(math.sqrt(model.get_size()[1]))
    patch_size = (size, size)

    for img in img_list:
        start = time.time()

        img_array = load_image(img)
        patches, i, j = crop(img_array, patch_size)
        # print("Cropped finished:", len(patches), i, j)
        logging.info("Training model using image: "+img)
        model.train(patches, i*j)
        stop = time.time()
        logging.info("Training model time: "+str((stop-start)/60)+" mins")
    logging.info("Training Finished, saving model to disk, path:"+save_path)
    model.save(save_path)


def test(test_path, model, result_path):
    """
    测试
    :param test_path:
    :param model:
    :return:
    """
    logging.info("Start test...")

    img_list = glob.glob(test_path + './*.jpg')

    assert float.is_integer(math.sqrt(model.get_size()[1]))

    size = int(math.sqrt(model.get_size()[1]))
    patch_size = (size, size)
    json_str = ''
    for img in img_list:
        start = time.time()
        img_array = load_image(img)
        patches, i, j = crop(img_array, patch_size)

        logging.info("Testing image: "+img)
        h = model.test(patches, i*j)

        logging.info("Computing value of anomaly for image: "+img+" patches")
        error = model.compute_anomaly(h, patches)

        stop = time.time()
        logging.info("Testing and computing anomaly in: "+str((stop-start)/60) +" mins")
        json_str += json.dumps({os.path.split(img)[1]: list(error)})

        json_path = result_path + os.path.split(img)[1][:-4] + ".json"
        with open(json_path, 'w') as f:
            logging.info("Writing json file to disk....")
            json.dump(json_str, f)
            json_str = ''
    logging.info("Test Finished.")


def show(model):
    """
    display learned dicts
    :param model:
    :return:
    """
    size_dict, size_img = model.get_size()
    d = model.get_dict()
    image_size = int(math.sqrt(size_img))

    temp = np.zeros((4*image_size, 4*image_size), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            temp[i*image_size: (i+1)*image_size, j*image_size: (j+1)*image_size] = \
                np.reshape(d[i*4+j], (image_size, image_size))

    plt.imshow(temp, cmap='gray')
    plt.show()
