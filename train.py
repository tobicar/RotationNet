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

##
helpers.rotate_image_and_plot("data/val2017/000000013348.jpg", 45)

## read dataset from disk
batch_size = 32
img_height = 224
img_width = 224
data_dir = "data/part9/"
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=None,
    validation_split=0.15,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=None
)
##
rotation_angles = tf.random.uniform(shape=(len(train_ds),), minval=0, maxval=359, dtype=tf.int32)

##
rotation_angles_dataset = tf.data.Dataset.from_tensor_slices(rotation_angles)
train_ds_with_labels = tf.data.Dataset.zip((train_ds, rotation_angles_dataset))
##
@tf.function
def rotate_img(img, label):
    # Load the raw data from the file as a string
    #img = tf.io.read_file(file_path)
    rotated_image = tf.numpy_function(helpers.rotate_image_and_crop, [img, label], tf.float32)
    #img = helpers.rotate_image_and_crop(img, label)
    label_one_hot = tf.one_hot(label, 360)
    return rotated_image, label_one_hot
##
train_ds_with_labels = train_ds_with_labels.map(rotate_img)


##
train_ds_with_labels = train_ds_with_labels.batch(32)
##
plt.figure(figsize=(10, 10))
for images, labels in train_ds_with_labels.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy())
        plt.axis("off")

##
def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(tf.keras.backend.argmax(y_true), tf.keras.backend.argmax(y_pred))
    return tf.keras.backend.mean(tf.keras.backend.cast(K.abs(diff), tf.keras.backend.floatx()))

## build model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

classes = 360
inputs = tf.keras.Input(shape=input_shape)
preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
base_model = base_model(preprocess_layer)

head = tf.keras.layers.Flatten()(base_model)
head = tf.keras.layers.Dense(512)(head)
head = tf.keras.layers.BatchNormalization()(head)
head = tf.keras.layers.Dropout(0.2)(head)
output = tf.keras.layers.Dense(classes, activation='softmax', name="RotationNetHead")(head)

model = tf.keras.Model(inputs, output, name="RotationNet")

##
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
##
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', angle_error])
##
history = model.fit(train_ds_with_labels, epochs=5)
## predict image

img = tf.keras.utils.load_img("data/val2017/000000003156.jpg")
img_arr = tf.keras.utils.img_to_array(img)
rot_img = helpers.rotate_image_and_crop(img_arr, 45)
img_arr_resized = tf.image.resize(rot_img, [224, 224])
img_exp = np.expand_dims(img_arr_resized, axis=0)
##
pred = model.predict(img_exp)


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
