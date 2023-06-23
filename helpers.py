import math
import time
import keras.backend as K
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background

    Source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.

    Source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rotate_image_and_plot(img_path, rotation_angle=90):
    # image = tf.keras.preprocessing.image.load_img(img_path)
    image = tf.keras.utils.load_img(img_path)
    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.utils.img_to_array(image)
    rotated_img = rotate_image(img_array, rotation_angle)
    image_height = img_array.shape[0]
    image_width = img_array.shape[1]
    image_rotated_cropped = crop_around_center(rotated_img,
                                               *largest_rotated_rect(
                                                   image_width,
                                                   image_height,
                                                   math.radians(rotation_angle)
                                               )
                                               )

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(top=0.9, hspace=0.3)
    ax1 = plt.subplot(3, 1, 1)
    ax1.title.set_text("Original image")
    plt.imshow(image)
    ax2 = plt.subplot(3, 1, 2)
    ax2.title.set_text("Rotated image by " + str(rotation_angle) + " degree")
    plt.imshow(tf.keras.utils.array_to_img(rotated_img))
    ax3 = plt.subplot(3, 1, 3)
    ax3.title.set_text("Rotated and cropped image")
    plt.imshow(tf.keras.utils.array_to_img(image_rotated_cropped))


def rotate_image_and_plot_from_array(img_arr, rotation_angle=90):
    image_rotated_cropped = rotate_image_and_crop(img_arr, rotation_angle)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 1, 1)
    if len(img_arr.shape) != 3:
        plt.imshow(tf.keras.utils.array_to_img(np.expand_dims(img_arr, axis=-1)))
    else:
        plt.imshow(tf.keras.utils.array_to_img(img_arr))
    ax = plt.subplot(2, 1, 2)
    if len(image_rotated_cropped.shape) != 3:
        plt.imshow(tf.keras.utils.array_to_img(np.expand_dims(image_rotated_cropped, axis=-1)))
    else:
        plt.imshow(tf.keras.utils.array_to_img(image_rotated_cropped))


def rotate_image_and_crop(image, rotation_angle):
    # image = image.numpy() if tf.is_tensor(image) else image
    image_height = image.shape[0]
    image_width = image.shape[1]
    # rotation_angle = rotation_angle.numpy() if tf.is_tensor(rotation_angle) else rotation_angle
    # if isinstance(rotation_angle, float):
    #    rotation_angle = math.radians(rotation_angle)
    # image = np.array(image, dtype=np.uint8)  # Convert to NumPy array
    image_rotated = rotate_image(image, rotation_angle)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(rotation_angle)
        )
    )
    return tf.image.resize(image_rotated_cropped, [image_height, image_width])


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
