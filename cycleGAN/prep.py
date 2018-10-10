# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          prep
Author:             deepgray
--------------------------------------------
Description:
# TODO : 读取数据和标签----这里是GAN网络，考虑怎么做

--------------------------------------------
Date:               18-6-27
Change Activity:

--------------------------------------------
"""
import tensorflow as tf
import os
from skimage import io
from skimage import transform
import logging
import pickle


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
            im_list.append(patch)
    return im_list, i, j
#
#
# def make_patch(image_path, patch_size=(256, 256)):
#     """
#     生成小块图像
#     """
#     file_list = os.listdir(image_path)
#     for im in file_list:
#         image = load_image(os.path.join(image_path, im))
#         im_list, _, _ = crop(image, patch_size)
#         file = os.path.join(os.getcwd(), os.path.join('Data', 'patches'))
#         # 清除目录
#         old_file = os.listdir(file)
#         for old in old_file:
#             os.remove(old)
#         for i, patch in enumerate(im_list):
#             name = os.path.join(file, str(i) + '.jpg')
#             io.imsave(name, patch)
#         logging.info("cropping image: %s to patches....", im)
#     logging.info("finished cropping images into patches.")


def label_generate(patch_path, label_path):
    label_dict = dict()
    file_list = os.listdir(patch_path)
    for file in file_list:
        img_pre, _ = os.path.splitext(file)
        xml_file = os.path.join(label_path, img_pre+'.xml')
        try:
            f = open(xml_file, 'r')
            label_dict[file] = 1
            f.close()
        except IOError:
            label_dict[file] = 0
    with open(os.path.join(os.getcwd()+'/Data/label', 'label.pkl'), 'wb', pickle.HIGHEST_PROTOCOL) as f:
        pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)


def write_tfrecords(image_path):
    """
    写入TFRecord文件
    """
    file_list = os.listdir(image_path)
    dir_name = os.path.join(os.path.join(os.getcwd(), 'Data'), 'label')
    file = open(os.path.join(dir_name, 'label.pkl'), 'rb')
    labels = pickle.load(file)
    num = len(file_list)
    assert num == len(labels)
    destination = os.path.join(os.path.join(os.getcwd(), 'Data'), 'tfrecords')
    # 粗略计算4000张照片大约110M，因此写入5个TFRecord文件中足够了
    with tf.Session() as sess:
        for i in range(5):
            file_name = os.path.join(destination, 'patches-%.2d-of-%.2d.tfrecords' % (i, 5))
            writer = tf.python_io.TFRecordWriter(file_name)
            logging.info('making tfrecords:'+str(i))
            for j in range(num//5):
                array = load_image(os.path.join(image_path, file_list[i*j]))
                image_raw = array.tostring()
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[file_list[i*j]]]))
                    }))
                writer.write(example.SerializeToString())
            writer.close()
    logging.info("TFrecords Write completed......")


if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', level=logging.DEBUG)
    # patch_xml(os.getcwd()+'/Data/images', os.getcwd()+'/Data/train/truth')
    # label_generate(os.getcwd()+'/Data/patches/possible_positive', os.getcwd()+'/xml_label')
    write_tfrecords(os.getcwd()+'/Data/patches/possible_positive')