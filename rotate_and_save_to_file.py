##
from PIL import Image
import os
import tensorflow as tf
import helpers

# Pfad zum Ordner mit den Bildern
folder_path = "data/val2017"
output_folder_path = "data/val2017_rotated"

rotation_angles_train = tf.random.uniform(shape=(len(os.listdir(folder_path)),), minval=0, maxval=359, dtype=tf.int32, seed=123)
##

for filename,rotation_angle in zip(os.listdir(folder_path), rotation_angles_train):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Vollständiger Pfad zur aktuellen Datei
        file_path = os.path.join(folder_path, filename)

        # Vollständiger Pfad zur Ausgabedatei
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

        # Bild rotieren und speichern
        #image = Image.open(file_path)
        image = tf.keras.utils.load_img(file_path)
        print(rotation_angle.numpy())
        rot_image_arr = helpers.rotate_image_and_crop(tf.keras.utils.img_to_array(image), rotation_angle.numpy())
        rot_img = tf.keras.utils.array_to_img(rot_image_arr)
        rot_img.save(output_path)
