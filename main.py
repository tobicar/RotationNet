##
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import mnist
import keras.backend as K

from RotationNetGenerator import RotNetDataGenerator

##
# we don't need the labels indicating the digit value, so we only load the images
(X_train, _), (X_test, _) = mnist.load_data()

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# classes to predict
nb_classes = 360

nb_train_samples, img_rows, img_cols = X_train.shape
input_shape = (img_rows, img_cols)
nb_test_samples = X_test.shape[0]


## model definition

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(nb_classes, activation="softmax"))
##

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
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def binarize_images(x):
    """
    Convert images to range 0-1 and binarize them by making
    0 the values below 0.1 and 1 the values above 0.1.
    """
    x /= 255
    x[x >= 0.1] = 1
    x[x < 0.1] = 0
    return x
## model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[angle_error])

##
# training parameters
batch_size = 128
nb_epoch = 50
# callbacks

early_stopping = EarlyStopping(patience=2)
tensorboard = TensorBoard()

# training loop
model.fit_generator(
    RotNetDataGenerator(
        X_train,
        batch_size=batch_size,
        preprocess_func=binarize_images,
        shuffle=True
    ),
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=RotNetDataGenerator(
        X_test,
        batch_size=batch_size,
        preprocess_func=binarize_images
    ),
    nb_val_samples=nb_test_samples,
    verbose=1,
    callbacks=[early_stopping, tensorboard]
)




## rotate and predict image
# randomly rotate an image
original_img = X_test[0]
true_angle = np.random.randint(360)
rotated_img = rotate(original_img, true_angle)
print('True angle: ', true_angle)

# add dimensions to account for the batch size and channels,
rotated_img = rotated_img[np.newaxis, :, :, np.newaxis]
# convert to float
rotated_img = rotated_img.astype('float32')
# binarize image
rotated_img_bin = binarize_images(rotated_img)
# predict rotation angle
output = model.predict(rotated_img_bin)
predicted_angle = np.argmax(output)
print('Predicted angle: ', predicted_angle)
