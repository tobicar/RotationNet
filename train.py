##
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import mnist
import keras.backend as K
import scipy.ndimage as ndimage
from RotationNetGenerator import RotNetDataGenerator
import math
import matplotlib.pyplot as plt
from PIL import Image
import helpers

##
helpers.rotate_image_and_plot("image.jpg",45)
