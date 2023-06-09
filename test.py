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
            lrr_width, lrr_height = largest_rotated_rect(output_height, output_width, math.radians(rotation_degree))
            print(lrr_width, lrr_height)
            resized_image = tf.image.central_crop(image, float(lrr_height) / output_height)
            image = tf.image.resize(resized_image, [output_height, output_width],
                                    method=tf.image.ResizeMethod.BILINEAR)

    return image


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

    image = rotate_and_crop(image, height, width, rotation_degree, do_crop)

    # if height and width:
    # Resize the image to the specified height and width.
    #image = tf.expand_dims(image, 0)
    #image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    # image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    #image = tf.squeeze(image, [0])

    # image = tf.cast(image, tf.float32)
    # image = tf.multiply(image, 1/255.)
    #image = tf.subtract(image, 0.5)
    #image = tf.multiply(image, 2.0)

    return image, label_one_hot


def rotate_image_and_plot(img_path, rotation_angle=90, height=256, width=256):
    #image = tf.keras.preprocessing.image.load_img(img_path)
    image = tf.keras.utils.load_img(img_path)
    #img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.utils.img_to_array(image)
    rotated_img_cropped = preprocess_image(img_array, rotation_angle, height=256, width=256, rotation_degree=rotation_angle, do_crop=True)
    rotated_img = preprocess_image(img_array, rotation_angle, height=256, width=256,rotation_degree=rotation_angle, do_crop=False)

    plt.figure(figsize=(10,10))
    ax = plt.subplot(3, 1, 1)
    plt.imshow(image)
    ax = plt.subplot(3, 1, 2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(rotated_img_cropped[0]))
    ax = plt.subplot(3, 1, 3)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(rotated_img[0]))


rotate_image_and_plot("image.jpg", rotation_angle=15)


