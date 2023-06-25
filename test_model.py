##
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import helpers

## load model
#model_path = "models/street_view_100Epoch"
model_path = "models/coco_100Epoch"
model = tf.keras.models.load_model(model_path, custom_objects={"angle_error": helpers.angle_error}, compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', helpers.angle_error])
## rotate and predict image

#file_path = "data/street_view/008954_2.jpg"  # select filepath for prediction
file_path = "data/coco/000000084477.jpg"  # select filepath for prediction
rotation_angles = [80, 190, 280, 40]

## take picture and predict some random rotations of the image
img = tf.keras.utils.load_img(file_path)
img_arr = tf.keras.utils.img_to_array(img)
plt.figure()
for index, angle in enumerate(rotation_angles):
    rot_img = helpers.rotate_image_and_crop(img_arr, angle)
    img_arr_resized = tf.image.resize(rot_img, [224, 224])
    img_exp = np.expand_dims(img_arr_resized, axis=0)
    pred = model.predict(img_exp)
    ax = plt.subplot(1, len(rotation_angles), index + 1)
    plt.imshow(img_arr_resized.numpy().astype("uint8"))
    plt.title(f"Label: {angle} Prediction: {pred[0].argmax()}")
    plt.axis("off")

