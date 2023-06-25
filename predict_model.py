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
model_path = "models/street_view_100Epoch"
model = tf.keras.models.load_model(model_path, custom_objects={"angle_error": helpers.angle_error}, compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', helpers.angle_error])
## rotate and predict image

file_path = "data/part9/008954_2.jpg"  # select filepath for prediction
file_path = "/Users/tobias/Downloads/bild_2.jpg"
rotation_angle = 180  # specifiy rotation angle
# randomly rotate an image
img = tf.keras.utils.load_img(file_path)
img_arr = tf.keras.utils.img_to_array(img)
rot_img = helpers.rotate_image_and_crop(img_arr, rotation_angle)
img_arr_resized = tf.image.resize(rot_img, [224, 224])
img_exp = np.expand_dims(img_arr_resized, axis=0)
pred = model.predict(img_exp)
plt.imshow(img_arr_resized.numpy().astype('uint8'))
plt.title(f"Original angle: {rotation_angle} - Predicted angle: {pred.argmax()}")
