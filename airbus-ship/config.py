import os

BATCH_SIZE = 2
EDGE_CROP = 16
NB_EPOCHS = 3
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = None

# downsampling in preprocessing
IMG_SCALING = (1, 1)

# number of validation images to use
VALID_IMG_COUNT = 2

# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 1
AUGMENT_BRIGHTNESS = False

# image directory
ship_dir = '/media/ax/文件/contest/airbus-ship'
train_image_dir = os.path.join(ship_dir, 'train')
# test_image_dir = os.path.join(ship_dir, 'test')
test_image_dir = os.path.join(ship_dir, 'test')
label_path = os.path.join(ship_dir, "train_ship_segmentations.csv")
