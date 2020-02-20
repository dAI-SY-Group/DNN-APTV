# Copyright 2020 TU Ilmenau. All Rights Reserved.
#
# Code for the Particle regression baseline for the
# On the use of a cascaded convolutional neural network for three-dimensional flow measurements using astigmatic PTV task.

# ==============================================================================





import math
import subprocess

import pandas as pd
from keras.callbacks import Callback
from keras_preprocessing.image import DataFrameIterator
import keras.backend as K


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except:
        return 'none'


class MultiInputDataFrameIterator(DataFrameIterator):

    def __init__(self, dataframe, directory=None, image_data_generator=None, x_col="filename", y_col="pos_z",
                 target_size=(256, 256), color_mode='grayscale', classes=None, class_mode='categorical', batch_size=32,
                 shuffle=True, seed=None, data_format='channels_last', save_to_dir=None, save_prefix='',samplewise_center=True,
                 samplewise_std_normalization=True,save_format='png', subset=None, interpolation='nearest', dtype='float32',
                 num_add_cols=0):
        super().__init__(dataframe, directory=directory, image_data_generator=image_data_generator, x_col=x_col, y_col=y_col,
                        target_size=target_size, color_mode=color_mode, classes=classes,
                         class_mode=class_mode, batch_size=batch_size, shuffle=shuffle,
                         seed=seed, data_format=data_format, save_to_dir=save_to_dir, save_prefix=save_prefix,
                         save_format=save_format, subset=subset, interpolation=interpolation,
                         dtype=dtype)
        self.num_add_cols = num_add_cols

    def _get_batches_of_transformed_samples(self, index_array):
        image, data = super()._get_batches_of_transformed_samples(index_array)
        in_data = [image]
        for i in range(self.num_add_cols):
            in_data.append(data[:, i + 1])
        return in_data, data[:, 0]

def get_class_form_data(csv_path, y_col='class'):
    df = pd.read_csv(csv_path)
    return df[y_col].values


def multi_input_data_gen(image_gen, csv_path, image_dir, batch_size=32, x_col='filename', y_col='pos_z', add_cols=None,
                         color_mode='grayscale', target_size=(221, 221), seed=None, shuffle=True, filter_border=None,filter_range=None):
    """
    Create an generator for multiple inputs where the additional inputs will be appended to image input

    :type filter_border: int
    :type filter_range: int
    :param filter_range: remove all test sample that are not in range e.g: -65-->25
    :param filter_border: remove all samples that are within a specified border e.g.: 100
    :param shuffle:
    :param csv_path: path to the csv file
    :param batch_size: batch size
    :param image_dir: path to the image directory
    :param x_col: name of the column containing the filename
    :param y_col: name of the column containing the pos_z/class
    :param add_cols: list of additional columns use for input. Default: ["posx", "posy"].
    :param color_mode: one of "grayscale", "grayscale". Default: "grayscale".
    :param target_size: image size as tuple. Default: (180, 180)
    :param seed: seed for random batch
    :type image_gen: ImageDataGenerator
    """
    if add_cols is None:
        add_cols = ['posx', 'posy']
    all_add_cols = add_cols.copy()
    all_add_cols.insert(0, y_col)
    df = pd.read_csv(csv_path)
    if filter_border is not None:
        is_in_border = (df['posx'] < filter_border) | (df['posy'] < filter_border) | ((2560 - df[
            'posx']) < filter_border) | ((2175- df['posy']) < filter_border)
        df = df[~is_in_border]
    if filter_range is not None:
        is_in_range =(df['pos_z'].astype(float) >=0) & (df['pos_z'].astype(float) <=60)
        df = df[is_in_range]

    return MultiInputDataFrameIterator(df, image_data_generator=image_gen,
                                       data_format=image_gen.data_format,
                                       x_col=x_col, y_col=all_add_cols, directory=image_dir,
                                       class_mode='raw', batch_size=batch_size, color_mode=color_mode,
                                       target_size=target_size, seed=seed, num_add_cols=len(add_cols), shuffle=shuffle)


class LearningRateExponentialDecay(Callback):
    def __init__(self, decay_rate, decay_steps):
        super().__init__()
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.global_step = 0

    def on_batch_begin(self, batch, logs=None):
        actual_lr = float(K.get_value(self.model.optimizer.lr))
        decayed_learning_rate = actual_lr * math.pow(self.decay_rate, math.floor(self.global_step / self.decay_steps))
        K.set_value(self.model.optimizer.lr, decayed_learning_rate)
        self.global_step += 1
