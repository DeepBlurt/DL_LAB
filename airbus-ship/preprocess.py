import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from rle_decode import masks_as_image
from config import *


def read_mask(csv_path, train_dir):
    masks = pd.read_csv(csv_path)
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(
        lambda x: 1.0 if x > 0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

    # some files are too small/corrupt
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(
        lambda c_img_id: os.stat(os.path.join(train_dir, c_img_id)).st_size / 1024)

    # keep only 50kb files
    unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]

    # unique_img_ids['file_size_kb'].hist()
    masks.drop(['ships'], axis=1, inplace=True)
    # unique_img_ids.sample(5)
    return masks, unique_img_ids


def split(csv_path, train_dir):
    masks, unique_img_ids = read_mask(csv_path, train_dir)
    train_ids, valid_ids = train_test_split(unique_img_ids, test_size=0.3,
                                            stratify=unique_img_ids['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    print(train_df.shape[0], 'training masks')
    print(valid_df.shape[0], 'validation masks')
    return train_df, valid_df


def sample_ships(in_df, base_rep_val=1500):
    if in_df['ships'].values[0] == 0:
        return in_df.sample(base_rep_val // 3)  # even more strongly undersample no ships
    else:
        return in_df.sample(base_rep_val, replace=(in_df.shape[0] < base_rep_val))


def under_sample(csv_path=label_path, train_dir=train_image_dir):
    train_df, _ = split(csv_path, train_dir)
    train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x + 1) // 2).clip(0, 7)
    balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)
    # balanced_train_df['ships'].hist(bins=np.arange(10))

    return balanced_train_df


# decode rle
# generator
def make_image_gen(in_df, train_dir=train_image_dir, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


def augment_config():
    dg_args = dict(featurewise_center=False,
                   samplewise_center=False,
                   rotation_range=15,
                   width_shift_range=0.1,
                   height_shift_range=0.1,
                   shear_range=0.01,
                   zoom_range=[0.9, 1.25],
                   horizontal_flip=True,
                   vertical_flip=True,
                   fill_mode = 'reflect',
                   data_format = 'channels_last')
    # brightness can be problematic since it seems to change the labels differently from the images
    if AUGMENT_BRIGHTNESS:
        dg_args[' brightness_range'] = [0.5, 1.5]
    image_gen = ImageDataGenerator(**dg_args)

    if AUGMENT_BRIGHTNESS:
        dg_args.pop('brightness_range')
    label_gen = ImageDataGenerator(**dg_args)
    return image_gen, label_gen


def create_aug_gen(in_gen, seed=None):
    image_gen, label_gen = augment_config()
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)
