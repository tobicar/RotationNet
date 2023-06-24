##
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import mnist
import keras.backend as K
import scipy.ndimage as ndimage
import math
import matplotlib.pyplot as plt
from PIL import Image
import helpers
import os

## read dataset from disk -> not needed anymore
data_dir = "data/val2017_rotated"
# list with image paths
image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]

# create list of labels with help of image_path
labels = []
for image_path in image_paths:
    # extract angle from filename
    angle = os.path.basename(image_path).split("_")[2].split(".")[0]
    labels.append(int(angle))

##
batch_size = 32
img_height = 224
img_width = 224
validation_split = 0.2

##
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "val2017_rotated",
    labels="inferred",
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    validation_split=validation_split,
    subset="both",
    shuffle=True,
    seed=42,
)
##
# one-hot encoding
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=360)))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=360)))
##
input_shape = (224, 224, 3)
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = True  #freeze layers of the backbone model and only train custom head

classes = 360
inputs = tf.keras.Input(shape=input_shape)
preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
base_model = base_model(preprocess_layer)

head = tf.keras.layers.GlobalAvgPool2D()(base_model)
#head = tf.keras.layers.GlobalAvgPool2D()(base_model)
#head = tf.keras.layers.Dense(512)(head)
#head = tf.keras.layers.BatchNormalization()(head)
head = tf.keras.layers.Dropout(0.4)(head)
output = tf.keras.layers.Dense(classes, activation='softmax', name="RotationNetHead")(head)

model = tf.keras.Model(inputs, output, name="RotationNet")
## plot structure of the model to file, if wanted
tf.keras.utils.plot_model(model, to_file="model_rotated_coco_dataset.png", show_shapes=True)
## compile model with adam optimizer and categorical_crossentropy (labels are one-hot encoded); use custom angle_error metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', helpers.angle_error])
##
history = model.fit(train_ds, validation_data= val_ds, epochs=100)
##
