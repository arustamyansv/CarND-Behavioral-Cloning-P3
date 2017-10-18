"""
File contains functions for data augmentation and statistics collection
"""

import cv2
import csv
import numpy as np
import math
import random
import os
from sklearn.utils import shuffle

def load_driving_log(path):
    """Load drivng log from the file system"""
    with open(os.path.join(path, 'driving_log.csv')) as f:
        log = []

        for line in csv.reader(f):
            line[0] = line[0].strip()
            line[1] = line[1].strip()
            line[2] = line[2].strip()
            line[3] = float(line[3])

            log.append(line)

    return log

def classify_angle(angle):
    """Function return classification index for angle"""
    func = 'ceil' if angle > 0 else 'floor'
    func = getattr(math, func)

    return func(angle/.1)

def collect_statistics(meta):
    """Function will get statistics for data distribution"""
    stats = {}

    for row in meta:
        angle = row[3]

        idx = classify_angle(angle)

        if idx not in stats:
            stats[idx] = 0

        stats[idx] += 1
    return stats

def redistribute(stats, meta, redist_limits):
    """Method that will leave at max 'dist_limit' number of samples for each angle class"""

    picked_stats = {key:0 for key in stats.keys()}
    redistributed_meta = []
    for row in shuffle(meta):
        angle = row[3]

        idx = classify_angle(angle)
        if idx in redist_limits and picked_stats[idx] >= redist_limits[idx]:
            continue

        picked_stats[idx] += 1
        redistributed_meta.append(row)

    return redistributed_meta

def side_cameras_augmentation(meta, corrections):
    """Parse meta and repopulate it with examples from left and right cameras"""
    result = []
    corrections = [0] + corrections

    for row in meta:
        angle = row[3]

        # process each of 3 rows from cvs for center, left and right images
        for i in range(3):
            # put proper path as first column in the info
            row[0] = row[i]

            # calculate correction rate
            corrected_angle = angle + corrections[i]
            if corrections[i] != 0:
                if corrected_angle < -1: corrected_angle = -1
                if corrected_angle >  1: corrected_angle = 1

            result.append(row)

    return result

def augmentate(image, angle, modes):
    """Function accumulates all data augmentaion techniques utilised in the project."""

    # select modes with some probability
    modes = [mode for mode in modes if random.randint(0, 1)]

    if 'flip' in modes:
        image, angle = flip(image, angle)

    if 'brightness' in modes:
        image = random_brightness(image)

    if 'shifting' in modes:
        image, angle = random_shifting(image, angle)

    return image, angle

def flip(image, angle):
    """Run random flipping"""

    image = cv2.flip(image, 1)
    angle *= -1

    return image, angle

def random_brightness(image):
    """Apply random brightness transformation so simulate different weather conditions"""

    # define random brightness factor
    bright_factor = np.random.uniform(0.3, 1.3)
    image = image.astype('float64')
    image[:,:,0] *= bright_factor

    # make sure we will not get value > 255
    image[:,:,0][image[:,:,0]>255] = 255

    image = image.astype('uint8')

    return image

def random_shifting(image, angle):
    """
    Shifts image randomly by x and y axis with maximum change of 15 pixels.
    In case we are shifting by x we will change steering angle accordingly by factor of 0.005 per pixel
    """

    max_shift = 10
    shift_y = random.randint(-max_shift, max_shift)
    shift_x = random.randint(-max_shift, max_shift)

    # adjust angle according to transformation. Will take 0.04 per pixel shifted
    angle += 0.04*shift_x

    M = np.float32([
        [1, 0, shift_y],
        [0, 1, shift_x]
    ])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return image, angle

def normalize(image, modes):
    """Main normalization method that run all normalization staff"""

    # crop image by height here. Removing unnecessary data from the top and bottom of the image
    if 'crop' in modes:
        image = image[modes['crop'][0]:modes['crop'][1],:]

    if 'gaussian' in modes:
        image = cv2.GaussianBlur(image, modes['gaussian'], 0)

    # resize image - we still have 320x90 px image. It will take much memory to process. Resize it for further processing
    if 'resize' in modes:
        image = cv2.resize(image, modes['resize'], interpolation = cv2.INTER_AREA)

    # cast to YUV as per NVIDIA paper
    if 'color_schema' in modes:
        image = cv2.cvtColor(image, modes['color_schema'])

    # histogram equalization
    if 'histogram' in modes:
        image[:,:,0] = cv2.equalizeHist(image[:,:,0])

    return image

gen_stats = {}

def generator(meta, CONFIG, stats=None, augment=False):
    """
    Reads images from file system, run augmentation and normalization process.
    Infinit loop.
    """
    # if we need to augment - get images from side cameras
    if augment:
        meta = side_cameras_augmentation(meta, CONFIG['augmentation']['side_cameras_corrections'])

    global gen_stats

    X_buff, y_buff = [], []
    while True:
        meta = shuffle(meta)

        # if we specified distribution limit run redistribution procedure
        if augment and CONFIG['redist_limits'] is not None:
            meta = redistribute(stats, meta, CONFIG['redist_limits'])

        for row in meta:
            fpath, angle = row[0], row[3]

            # check if buffer is full enough - yield data and flush buffers
            if len(X_buff) == CONFIG['batch_size']:
                yield np.array(X_buff, dtype='float32'), np.array(y_buff, dtype='float32')
                X_buff, y_buff = [], []

            fpath = os.path.join(CONFIG['data_path'], fpath)

            # missed some images so it's to make sure i'll not trap into error
            if not os.path.isfile(fpath):
                continue

            image = cv2.imread(fpath)

            image = normalize(image, CONFIG['normalization'])

            if augment:
                image, angle = augmentate(image, angle, CONFIG['augmentation']['modes'])

            angle_class = classify_angle(angle)
            if angle_class not in gen_stats:
                gen_stats[angle_class] = 0

            gen_stats[angle_class] += 1

            X_buff.append(image)
            y_buff.append(angle)