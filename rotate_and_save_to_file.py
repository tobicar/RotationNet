##
from PIL import Image
import os
import tensorflow as tf
import helpers

# path to folder with images
folder_path = "data/coco"
output_folder_path = "data/coco_rotated"

rotation_angles_train = tf.random.uniform(shape=(len(os.listdir(folder_path)),), minval=0, maxval=359, dtype=tf.int32, seed=123)
##

for filename,rotation_angle in zip(os.listdir(folder_path), rotation_angles_train):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # complete path to actual file
        file_path = os.path.join(folder_path, filename)

        # complete path to output file
        if filename.endswith(".jpg"):
            new_filename = filename.split(".")[0] + "_angle_" + str(rotation_angle.numpy()) + ".jpg"
        if filename.endswith(".png"):
            new_filename = filename.split(".")[0] + "_angle_" + str(rotation_angle.numpy()) + ".png"
        angle_subdir = str(rotation_angle.numpy())
        subfolder_path = os.path.join(output_folder_path, angle_subdir)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print("Created folder:", subfolder_path)
        else:
            print("Folder already exists:", subfolder_path)
        output_path = os.path.join(subfolder_path, new_filename)
        print(output_path)

        # rotate image and save
        image = tf.keras.utils.load_img(file_path)
        print(rotation_angle.numpy())
        rot_image_arr = helpers.rotate_image_and_crop(tf.keras.utils.img_to_array(image), rotation_angle.numpy())
        rot_img = tf.keras.utils.array_to_img(rot_image_arr)
        rot_img.save(output_path)
