#!/usr/bin/env python

import cv2
import numpy as np
import math

from pprint import pprint
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianDropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# custom helpers for this project
from helpers import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', './data', 'Define folder with training data')
flags.DEFINE_string('lm', 'train', 'Define if we need to train from scratch or finetune the model. Could be "train" or "fine-tune"')

CONFIG = {
    # learning mode is one of the following ('train', 'fine-tune')
    'learning_mode': FLAGS.lm,
    'data_path': FLAGS.f,

    # 'redist_limits': None,
    'redist_limits': {
#         -10: 0,
#         -9: 0,
#         -8: 0,
#         -7: 0,
#         -6: 0,
         -2: 300,
         -1: 300,
         0:  300,
         1:  300,
         2:  300,
#         6: 300,
#         7: 300,
#         8: 300,
#         9: 300,
#         10: 300
     },

    'batch_size': 256,

    'normalization': {
        'crop': (50, 140),
        'resize': (200, 66),
        'color_schema': cv2.COLOR_BGR2YUV,
#         'gaussian': (3,3),
#         'histogram': True,
    },

    'augmentation': {
        'side_cameras_corrections': [.25, -.25],
        'modes': [
            'flip',
            # 'brightness',
            'shifting'
        ]
    },

    'input_shape': [66, 200],
    'samples_per_epoch': 256*250,
    'epoches': 25
}

# compose log path
print ('Loading Driving Log ...')
meta = load_driving_log(CONFIG['data_path'])

print ('Initial data distribution:')
stats = collect_statistics(meta)
pprint(stats)

# split data for train and validation set
train_meta, valid_meta = train_test_split(meta, test_size = .2)

# define generators
train_gen = generator(train_meta, CONFIG, stats=stats, augment=True)
valid_gen = generator(valid_meta, CONFIG)

# if learning mode is fine tune try to load previous model
model = None
if CONFIG['learning_mode'] == 'fine-tune':
    if os.path.isfile('model.h5'):
        print ('Loading saved model...')
        model = load_model('model.h5')

if model is None:
    # Nvidia architecture
    model = Sequential([
        Lambda(lambda x: x / 255.0 - .5, input_shape=(CONFIG['input_shape'] + [3])),

        Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu', W_regularizer=l2(0.001)),
        Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu', W_regularizer=l2(0.001)),
        Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu', W_regularizer=l2(0.001)),

        Convolution2D(64, 3, 3, activation='elu', W_regularizer=l2(0.001)),
        Convolution2D(64, 3, 3, activation='elu', W_regularizer=l2(0.001)),

        Flatten(),

        Dense(100, activation='elu', W_regularizer=l2(0.001)),
        Dense(50, activation='elu', W_regularizer=l2(0.001)),
        Dense(10, activation='elu', W_regularizer=l2(0.001)),
        Dense(1)
    ])

model.compile(loss='mse', optimizer='adam')
model.summary()

model.fit_generator(
    generator = train_gen,
    samples_per_epoch = CONFIG['samples_per_epoch'],
    nb_epoch = CONFIG['epoches'],
    validation_data = valid_gen,
    nb_val_samples = len(valid_meta)
)
pprint(gen_stats)

model.save('model.h5')