# Rotation Net

Estimation of image rotation angle with convolutional neural networks

# Installation: 
- link to github repository: https://github.com/tobicar/RotationNet
- created with pycharm IDE
- parts of the code better executable with extension [pycharm cell mode](https://plugins.jetbrains.com/plugin/7858-pycharm-cell-mode)

## links to the used datasets:
- https://drive.google.com/drive/folders/15beJl1wzXnVS1MxS23P35KZ2HMl9YX1D?usp=sharing
- https://drive.google.com/drive/folders/1krabuviNRljeFoleDO_yUBg43UvG8Xce?usp=sharing


## structure of the project:

### files:

- [train_coco_dataset.py](train_coco_dataset.py)
  - This script can be used to train the model on the COCO dataset. The trained model is saved in the [models](models) folder and the history in the [models_history](models_history) folder.
- [train_street_view_dataset.py](train_street_view_dataset.py)
  - This script can be used to train the model on the Street View dataset. The trained model is saved in the [models](models) folder and the history in the [models_history](models_history) folder.
- [predict_model.py](predict_model.py)
  - Here a trained model can be loaded and a prediction of an image can be performed. The image can be loaded via a FilePath and the rotation angle for the previously performed rotation can be specified.
- [evaluate_dataset.py](evaluate_dataset.py)
  - In this script, a trained model can be loaded and an evaluation of the validation dataset can be performed with the model.
- [create_history_plot_from_file.py](create_history_plot_from_file.py)
  - This script reads the saved history of a trained model and generates a loss and an angle error plot.
- [helpers.py](helpers.py)
  - This script contains helper functions necessary for training, evaluation and testing.  These are on the one hand image rotation functions, custom loss functions and plot functions.
- [test_model.py](test_model.py)
  - Script to test different rotations of an image with a trained model.
- Files that are no longer used:
  - [rotate_and_save_to_file.py](rotate_and_save_to_file.py)
    - File loads images from a specified folder and stores them in a new folder randomly rotated. The target structure is such that one folder is created per rotation angle.
  - [generate_rotated_file_structure.py](generate_rotated_file_structure.py)
    - File loads images from a folder where the images are already rotated without structure. The images are moved to a destination folder and a folder is created for each rotation angle.
  - [train_rotated_coco_dateset.py](train_rotated_coco_dateset.py)
    - File to train a model directly on rotated images.

### directories:

- data
  - Must be created by downloading the images from Google Drive and then pushing them into the Models folder
- [models](models)
  - Contains the two models trained over 100 epochs
- [models_history](models_history)
  - Contains the histories in Numpy format of the trained models