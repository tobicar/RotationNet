##
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
#history_path = "models_history/coco_100Epoch.npy"
history_path = "models_history/street_view_100Epoch.npy"
history = np.load(history_path, allow_pickle=True).item()

## plot the training history (loss)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
## plot the training angle error
plt.plot(history['angle_error'])
plt.plot(history['val_angle_error'])
plt.title('model angle error')
plt.ylabel('angle error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()