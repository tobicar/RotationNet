##
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import mnist
import keras.backend as K
import scipy.ndimage as ndimage
from RotationNetGenerator import RotNetDataGenerator
import math
import matplotlib.pyplot as plt
from PIL import Image
import helpers
import os
##
data_dir = "val2017/*"
list_ds = tf.data.Dataset.list_files(data_dir, shuffle=False)
image_count = len(list_ds)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

#val_size = int(image_count * 0.2)
#train_ds = list_ds.skip(val_size)
#val_ds = list_ds.take(val_size)

##
img_height = 256
img_width= 256

def decode_img(img,label):
  # Convert the compressed string to a 3D uint8 tensor
  img_arr = tf.keras.utils.load_img(img)
  img = tf.keras.utils.img_to_array(img_arr)

  #helpers.rotate_image_and_crop(img,label)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path, label):
  # Load the raw data from the file as a string
  #img = tf.io.read_file(file_path)
  img = decode_img(file_path, label)
  return label, img


##
rotation_angles = tf.random.uniform(shape=(len(list_ds),), minval=0, maxval=360, dtype=tf.int32)

##
rotation_angles_dataset = tf.data.Dataset.from_tensor_slices(rotation_angles)
train_ds = tf.data.Dataset.zip((list_ds,rotation_angles_dataset))
##
train_ds.map(process_path)