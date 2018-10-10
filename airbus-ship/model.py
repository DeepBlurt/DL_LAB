from keras import models, layers
from config import *
import keras.backend as K
from keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from skimage.io import imread
from skimage.morphology import binary_opening, disk
import gc
from rle_decode import *
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau

# Build U-Net model


def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)


def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_true_positive_rate'] for mh in loss_history]),
                 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')

    _ = ax3.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                 'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')

    _ = ax4.plot(epich, np.concatenate(
        [mh.history['dice_coef'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_dice_coef'] for mh in loss_history]),
                 'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')
    plt.savefig('loss_history.png')


class Unet(object):
    def __init__(self, input_shape):
        if UPSAMPLE_MODE == 'DECONV':
            self.upsample = upsample_conv
        else:
            self.upsample = upsample_simple

        self.input_img = layers.Input(input_shape, name='RGB_Input')
        self.pp_in_layer = self.input_img
        if NET_SCALING is not None:
            self.pp_in_layer = layers.AvgPool2D(NET_SCALING)(self.pp_in_layer)

        self.pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(self.pp_in_layer)
        self.pp_in_layer = layers.BatchNormalization()(self.pp_in_layer)

        self.c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.pp_in_layer)
        self.c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.c1)
        self.p1 = layers.MaxPooling2D((2, 2))(self.c1)

        self.c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.p1)
        self.c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.c2)
        self.p2 = layers.MaxPooling2D((2, 2))(self.c2)

        self.c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.p2)
        self.c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.c3)
        self.p3 = layers.MaxPooling2D((2, 2))(self.c3)

        self.c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(self.p3)
        self.c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(self.c4)
        self.p4 = layers.MaxPooling2D(pool_size=(2, 2))(self.c4)

        self.c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(self.p4)
        self.c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(self.c5)

        self.u6 = self.upsample(64, (2, 2), strides=(2, 2), padding='same')(self.c5)
        self.u6 = layers.concatenate([self.u6, self.c4])
        self.c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(self.u6)
        self.c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(self.c6)

        self.u7 = self.upsample(32, (2, 2), strides=(2, 2), padding='same')(self.c6)
        self.u7 = layers.concatenate([self.u7, self.c3])
        self.c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.u7)
        self.c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.c7)

        self.u8 = self.upsample(16, (2, 2), strides=(2, 2), padding='same')(self.c7)
        self.u8 = layers.concatenate([self.u8, self.c2])
        self.c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.u8)
        self.c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.c8)

        self.u9 = self.upsample(8, (2, 2), strides=(2, 2), padding='same')(self.c8)
        self.u9 = layers.concatenate([self.u9, self.c1], axis=3)
        self.c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.u9)
        self.c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.c9)

        self.d = layers.Conv2D(1, (1, 1), activation='sigmoid')(self.c9)
        self.d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(self.d)
        self.d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(self.d)

        if NET_SCALING is not None:
            self.d = layers.UpSampling2D(NET_SCALING)(self.d)

        self.seg_model = models.Model(inputs=[self.input_img], outputs=[self.d])
        self.seg_model.compile(optimizer=Adam(1e-4, decay=1e-6),
                               loss=dice_p_bce,
                               metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
        print(self.seg_model.summary())
        self.weight_path = "{}_weights.best.hdf5".format('seg_model')

        self.checkpoint = ModelCheckpoint(self.weight_path, monitor='val_dice_coef', verbose=1,
                                          save_best_only=True, mode='max', save_weights_only=True)

        self.reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                                patience=3,
                                                verbose=1, mode='max',
                                                epsilon=0.0001, cooldown=2, min_lr=1e-6)

        self.early = EarlyStopping(monitor="val_dice_coef",
                                   mode="max",
                                   patience=20)

    def train(self, balanced_train_df, valid_x, valid_y, create_aug_gen, make_image_gen):
        if os.path.exists(self.weight_path):
            self.load_weights(self.weight_path)
        callbacks_list = [self.checkpoint, self.early, self.reduceLROnPlat]
        step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0] // BATCH_SIZE)
        aug_gen = create_aug_gen(make_image_gen(balanced_train_df))

        # the generator is not very thread safe
        loss_history = [self.seg_model.fit_generator(aug_gen,
                                                     steps_per_epoch=step_count,
                                                     epochs=NB_EPOCHS,
                                                     validation_data=(valid_x, valid_y),
                                                     callbacks=callbacks_list,
                                                     workers=1)]
        self.seg_model.load_weights(self.weight_path)
        self.seg_model.save('seg_model.h5')
        pred_y = self.seg_model.predict(valid_x)
        print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())
        return loss_history

    def load_weights(self, path):
        self.seg_model.load_weights(path)

    def test(self, test_path=test_image_dir):
        gc.enable()
        self.load_weights("{}_weights.best.hdf5".format('seg_model'))

        # Prepare Full Resolution Model
        if IMG_SCALING is not None:
            fullres_model = models.Sequential()
            fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape=(None, None, 3)))
            fullres_model.add(self.seg_model)
            fullres_model.add(layers.UpSampling2D(IMG_SCALING))
        else:
            fullres_model = self.seg_model
        fullres_model.save('fullres_model.h5')

        test_paths = os.listdir(test_path)
        print(len(test_paths), 'test images found')
        fig, m_axs = plt.subplots(8, 2, figsize=(10, 40))
        [c_ax.axis('off') for c_ax in m_axs.flatten()]
        for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
            c_path = os.path.join(test_path, c_img_name)
            c_img = imread(c_path)
            first_img = np.expand_dims(c_img, 0) / 255.0
            first_seg = fullres_model.predict(first_img)
            ax1.imshow(first_img[0])
            ax1.set_title('Image')
            ax2.imshow(first_seg[0, :, :, 0], vmin=0, vmax=1)
            ax2.set_title('Prediction')
        fig.savefig('test_predictions.png')
        out_pred_rows = []

        for c_img_name in tqdm(test_paths):
            c_path = os.path.join(test_image_dir, c_img_name)
            c_img = imread(c_path)
            c_img = np.expand_dims(c_img, 0) / 255.0
            cur_seg = fullres_model.predict(c_img)[0]
            cur_seg = binary_opening(cur_seg > 0.5, np.expand_dims(disk(2), -1))
            cur_rles = multi_rle_encode(cur_seg)
            if len(cur_rles) > 0:
                for c_rle in cur_rles:
                    out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': c_rle}]
            else:
                out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': None}]
            gc.collect()
        submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
        submission_df.to_csv('submission_test.csv', index=False)
        submission_df.sample(3)

        submission_df['counts'] = submission_df.apply(
            lambda c_row: c_row['counts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)
        submission_df['counts'].hist()
