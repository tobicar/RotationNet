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

##
helpers.rotate_image_and_plot("image.jpg",45)

##
# we don't need the labels indicating the digit value, so we only load the images
(X_train, _), (X_test, _) = mnist.load_data()

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# classes to predict
nb_classes = 360

nb_train_samples, img_rows, img_cols = X_train.shape
input_shape = (img_rows, img_cols)
nb_test_samples = X_test.shape[0]
## test rotation of one image and plot it
helpers.rotate_image_and_plot_from_array(X_train[0], 180)
##
rotation_angles = np.random.randint(low=0, high=360, size=len(X_train))
labels = rotation_angles.astype(np.int32)
##
rotation_angles = tf.random.uniform(shape=(len(X_train),), minval=0, maxval=360, dtype=tf.int32)
##

def convert_label(label):
    return int(label)

##
def preprocess_image(image, label, height=224, width=224):
    # if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #rotation_degree = float(label)
    # Convert the label tensor to a Python integer
    #rotation_degree = tf.py_function(convert_label, [label], tf.int64)
    rotation_angle = tf.cast(label, dtype=tf.float32)
    image_rot_crop = helpers.rotate_image_and_crop(image, rotation_angle)
    resized_rotated_image = tf.image.resize(image_rot_crop, [28, 28])  # Resize the image
    resized_original_image = tf.image.resize(image, [28, 28])

    # if height and width:
    # Resize the image to the specified height and width.
    #image = tf.expand_dims(image, 0)
    #image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    # image = tf.image.resize_bilinear(image, [height, width], align_corners=False) method doesnt exist anymore

    # TODO: Maybe this is not needed? What does this do?
    #image = tf.squeeze(image, [0])

    # image = tf.cast(image, tf.float32)
    # image = tf.multiply(image, 1/255.)
    #image = tf.subtract(image, 0.5)
    #image = tf.multiply(image, 2.0)

    return resized_rotated_image, resized_original_image, rotation_angle

##
dataset = tf.data.Dataset.from_tensor_slices((X_train, rotation_angles))
##dataset = dataset.map(lambda x, y: print(x))
##
dataset = dataset.map(preprocess_image)
dataset = dataset.shuffle(len(X_train))
dataset = dataset.batch(32)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
##
def print_images(dataset, num_examples=1):
    dataset_iter = iter(dataset)

    fig = plt.figure(figsize=(10, 10))

    for i in range(num_examples):
        rotated_image, image, label = next(dataset_iter)
        image = image.numpy()
        rotated_image = rotated_image.numpy()

        ax1 = fig.add_subplot(3, 3, i * 2 + 1)
        ax1.imshow(image)
        ax1.set_title(f"Original - Label: {label}")

        ax2 = fig.add_subplot(3, 3, i * 2 + 2)
        ax2.imshow(rotated_image)
        ax2.set_title(f"Rotated")

    plt.tight_layout()
    plt.show()
##
print_images(dataset,num_examples=9)
