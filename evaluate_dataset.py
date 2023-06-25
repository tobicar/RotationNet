##
import tensorflow as tf
import numpy as np
import keras.backend as K
import scipy.ndimage as ndimage
import math
import matplotlib.pyplot as plt
from PIL import Image
import helpers
## specifiy data path and path of the trained model
data_dir = "data/street_view/"
model_path = "models/street_view_100Epoch"
## read dataset from disk
img_height = 224
img_width = 224
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=None,
    validation_split=0.15,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=None
)
## generate uniform distribution of rotation angles for train and val set
rotation_angles_train = tf.random.uniform(shape=(len(train_ds),), minval=0, maxval=359, dtype=tf.int32, seed=132)
rotation_angles_val = tf.random.uniform(shape=(len(val_ds),), minval=0, maxval=359, dtype=tf.int32, seed=132)
## function for applying rotation to image of each dataset
@tf.function
def rotate_img(img, label):
    rotated_image = tf.numpy_function(helpers.rotate_image_and_crop, [img, label], tf.float32)
    label_one_hot = tf.one_hot(label, 360)
    #label_one_hot = tf.reshape(label_one_hot,(-1,360))
    return rotated_image, label_one_hot


## zip both datasets together for train and val sets
rotation_angles_dataset = tf.data.Dataset.from_tensor_slices(rotation_angles_train)
train_ds_with_labels = tf.data.Dataset.zip((train_ds, rotation_angles_dataset))

rotation_angles_dataset_val = tf.data.Dataset.from_tensor_slices(rotation_angles_val)
val_ds_with_labels = tf.data.Dataset.zip((val_ds, rotation_angles_dataset_val))
## apply preprocessing function (rotation) to both datasets
train_ds_with_labels = train_ds_with_labels.map(rotate_img)
val_ds_with_labels = val_ds_with_labels.map(rotate_img)
## batch the datasets
train_ds_with_labels = train_ds_with_labels.batch(64)
val_ds_with_labels = val_ds_with_labels.batch(64)
## load model
model = tf.keras.models.load_model(model_path, custom_objects={"angle_error": helpers.angle_error}, compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', helpers.angle_error])
##
model.evaluate(val_ds_with_labels)