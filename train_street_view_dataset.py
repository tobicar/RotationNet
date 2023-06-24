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

## read dataset from disk
batch_size = 32
img_height = 224
img_width = 224
data_dir = "part9/"
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
## build transfer learning model
input_shape = (224, 224, 3)
tf.keras.applications.resnet
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False  #freeze layers of the backbone model and only train custom head

classes = 360
inputs = tf.keras.Input(shape=input_shape)
preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
base_model = base_model(preprocess_layer)

head = tf.keras.layers.GlobalAvgPool2D()(base_model)
#head = tf.keras.layers.GlobalAvgPool2D()(base_model)
#head = tf.keras.layers.Dense(512)(head)
#head = tf.keras.layers.BatchNormalization()(head)
#head = tf.keras.layers.Dropout(0.2)(head)
output = tf.keras.layers.Dense(classes, activation='softmax', name="RotationNetHead")(head)

model = tf.keras.Model(inputs, output, name="RotationNet")
## plot structure of the model to file, if wanted
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
## compile model with adam optimizer and categorical_crossentropy (labels are one-hot encoded); use custom angle_error metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', helpers.angle_error])
##
history = model.fit(train_ds_with_labels, validation_data= val_ds_with_labels, epochs=10)
## plot the training history (loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
## plot the training angle error
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

