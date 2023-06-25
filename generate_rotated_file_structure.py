import os
import shutil
# file copies images from a folder structurized to another one
# imagesfilename structure in source folder "name_name_angle_"specificangle".jpg
# creates subfolder for each rotation class in destination folder

# path to folder with images
image_folder = "coco_rotated"

# create destination folder for sorted pictures
sorted_folder = "val2017_rotated_subfolder"
os.makedirs(sorted_folder, exist_ok=True)

# list all images in folder
image_files = os.listdir(image_folder)


for image_file in image_files:
    # extract angle from filename
    angle = image_file.split("_")[2]

    # create path to actual image
    image_path = os.path.join(image_folder, image_file)

    # create path to destination dir for the current angle
    angle_folder = os.path.join(sorted_folder, angle)
    os.makedirs(angle_folder, exist_ok=True)

    # move image in specific folder
    shutil.move(image_path, os.path.join(angle_folder, image_file))