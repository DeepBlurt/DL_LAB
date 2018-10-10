from preprocess import make_image_gen, split, create_aug_gen, under_sample
from config import *
from model import Unet, show_loss
import gc
import numpy as np
# # test code imports
# import pandas as pd
# from rle_decode import masks_as_image, multi_rle_encode
from skimage.util.montage import montage2d as montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
import matplotlib.pyplot as plt

# memory is tight
gc.enable()

# # test code
# masks = pd.read_csv(label_path)
# print(masks.shape[0], "mask found")
# print(masks['ImageId'].value_counts().shape[0])
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
# rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
# img_0 = masks_as_image(rle_0)
# ax1.imshow(img_0[:, :, 0])
# ax1.set_title('Image$_0$')
# rle_1 = multi_rle_encode(img_0)
# img_1 = masks_as_image(rle_1)
# ax2.imshow(img_1[:, :, 0])
# ax2.set_title('Image$_1$')
# print('Check Decoding->Encoding',
#       'RLE_0:', len(rle_0), '->',
#       'RLE_1:', len(rle_1))
# plt.show()
#
# masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
# unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
# unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
# unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
# # some files are too small/corrupt
# unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
#                                                                os.stat(os.path.join(train_image_dir,
#                                                                                     c_img_id)).st_size/1024)
# unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] # keep only 50kb files
# unique_img_ids['file_size_kb'].hist()
# masks.drop(['ships'], axis=1, inplace=True)
# print(unique_img_ids.sample(5))
# plt.show()


# train data check
# train_df, valid_df = split(csv_path=label_path, train_dir=train_image_dir)
balanced_df = under_sample(csv_path=label_path, train_dir=train_image_dir)
print("balanced_df shape0:", balanced_df.shape)
# balanced_df['ships'].hist(bins=np.arange(10))
# print(balanced_df.sample(10))
# plt.show()

train_gen = make_image_gen(balanced_df)
train_x, train_y = next(train_gen)
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())

# valid data check
valid_x, valid_y = next(make_image_gen(balanced_df, batch_size=VALID_IMG_COUNT))
print(valid_x.shape, valid_y.shape)

# augment data check
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
# ax1.imshow(montage_rgb(t_x), cmap='gray')
# ax1.set_title('images')
# ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')
# ax2.set_title('ships')
# plt.show()


model = Unet(t_x.shape[1:])
loss_history = model.train(balanced_train_df=balanced_df,
                           valid_x=valid_x, valid_y=valid_y,
                           make_image_gen=make_image_gen,
                           create_aug_gen=create_aug_gen)

show_loss(loss_history)
# model.test()

