##
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import mnist
import keras.backend as K
import scipy.ndimage as ndimage
from RotationNetGenerator import RotNetDataGenerator
import math
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from PIL import Image


##
def decode_image(img_path):
    """
    decoding and preprocessing of image
    :param img_path: path to image
    :return: image in shape (244,244,3)
    """
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    # img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


##
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

##
def rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop):
    """Rotate the given image with the given rotation degree and crop for the black edges if necessary
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        rotation_degree: The degree of rotation on the image.
        do_crop: Do cropping if it is True.

    Returns:
        A rotated image.
    """
    # Rotate the given image with the given rotation degree
    if rotation_degree != 0:
        image_pil = tf.keras.utils.array_to_img(image)
        #image_pil = tf.keras.preprocessing.image.array_to_img(image)
        image_pil = image_pil.rotate(rotation_degree, resample=Image.BILINEAR)
        # image = tfa.image.rotate(image, math.radians(rotation_degree), interpolation='BILINEAR')
        #image = tf.keras.preprocessing.image.img_to_array(image_pil)
        image = tf.keras.utils.img_to_array(image_pil)
        # image = tf.contrib.image.rotate(image, math.radians(rotation_degree), interpolation='BILINEAR')

        # Center crop to ommit black noise on the edges
        if do_crop == True:
            lrr_width, lrr_height = largest_rotated_rect(output_width, output_height, math.radians(rotation_degree))
            print(lrr_width, lrr_height)
            print("central cropped: " + str(float(lrr_height) / output_height))
            resized_image = tf.image.central_crop(image, float(lrr_height) / output_height)
            image = tf.image.resize(resized_image, [output_height, output_width],
                                    method=tf.image.ResizeMethod.BILINEAR)

    return image


## rotate one image and plot it
def rotate_image_and_plot(img_path, rotation_angle=90, height=256, width=256):
    #image = tf.keras.preprocessing.image.load_img(img_path)
    image = tf.keras.utils.load_img(img_path)
    #img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.utils.img_to_array(image)
    rotated_img_cropped = preprocess_image(img_array, rotation_angle, height=height, width=width, rotation_degree=rotation_angle, do_crop=True)
    rotated_img = preprocess_image(img_array, rotation_angle, height=height, width=width,rotation_degree=rotation_angle, do_crop=False)

    plt.figure(figsize=(10,10))
    ax = plt.subplot(4, 1, 1)
    plt.imshow(image)
    ax = plt.subplot(4, 1, 2)
    image_resized = tf.image.resize(img_array, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image_resized))
    ax = plt.subplot(4, 1, 3)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(rotated_img_cropped[0]))
    ax = plt.subplot(4, 1, 4)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(rotated_img[0]))



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

##
rotation_angles = np.random.randint(low=0, high=360, size=len(X_train))
labels = rotation_angles.astype(np.float32)

##
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360)


def preprocess_image(image, label_one_hot, height=224, width=224, rotation_degree=0, do_crop=False):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would cropt the central fraction of the
    input image.

    Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
    Returns:
    3-D float Tensor of prepared image.
    """

    # if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    # if central_fraction:
    #  image = tf.image.central_crop(image, central_fraction=central_fraction)
    print(image.shape)
    image_height = image.shape[0]
    image_width = image.shape[1]
    image = rotate_and_crop(image, image_height, image_width, rotation_degree, do_crop)

    # if height and width:
    # Resize the image to the specified height and width.
    #image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    # image = tf.image.resize_bilinear(image, [height, width], align_corners=False) method doesnt exist anymore

    # TODO: Maybe this is not needed? What does this do?
    #image = tf.squeeze(image, [0])

    # image = tf.cast(image, tf.float32)
    # image = tf.multiply(image, 1/255.)
    #image = tf.subtract(image, 0.5)
    #image = tf.multiply(image, 2.0)

    return image, label_one_hot


def apply_rotation(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


##
dataset = tf.data.Dataset.from_tensor_slices((X_train, labels))
dataset = dataset.map(apply_rotation)
dataset = dataset.map(lambda x, y: (preprocess_image(image=x, label_one_hot=y, rotation_degree=y, do_crop=True), y))
dataset = dataset.shuffle(len(X_train))
dataset = dataset.batch(32)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

## model definition

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(nb_classes, activation="softmax"))


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
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


def binarize_images(x):
    """
    Convert images to range 0-1 and binarize them by making
    0 the values below 0.1 and 1 the values above 0.1.
    """
    x /= 255
    x[x >= 0.1] = 1
    x[x < 0.1] = 0
    return x


## model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[angle_error])

##
# training parameters
batch_size = 128
nb_epoch = 50
# callbacks

early_stopping = EarlyStopping(patience=2)
tensorboard = TensorBoard()

# training loop
model.fit_generator(
    RotNetDataGenerator(
        X_train,
        batch_size=batch_size,
        preprocess_func=binarize_images,
        shuffle=True
    ),
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=RotNetDataGenerator(
        X_test,
        batch_size=batch_size,
        preprocess_func=binarize_images
    ),
    nb_val_samples=nb_test_samples,
    verbose=1,
    callbacks=[early_stopping, tensorboard]
)

## rotate and predict image
# randomly rotate an image
original_img = X_test[0]
true_angle = np.random.randint(360)
rotated_img = rotate(original_img, true_angle)
print('True angle: ', true_angle)

# add dimensions to account for the batch size and channels,
rotated_img = rotated_img[np.newaxis, :, :, np.newaxis]
# convert to float
rotated_img = rotated_img.astype('float32')
# binarize image
rotated_img_bin = binarize_images(rotated_img)
# predict rotation angle
output = model.predict(rotated_img_bin)
predicted_angle = np.argmax(output)
print('Predicted angle: ', predicted_angle)
