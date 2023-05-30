from keras.preprocessing.image import Iterator


class RotNetDataGenerator(Iterator):

    def __init__(self, input, batch_size=64,
                 preprocess_func=None, shuffle=False):

        self.images = input
        self.batch_size = batch_size
        self.input_shape = self.images.shape[1:]
        self.preprocess_func = preprocess_func
        self.shuffle = shuffle
        # add dimension if the images are greyscale
        if len(self.input_shape) == 2:
            self.input_shape = self.input_shape + (1,)
        N = self.images.shape[0]

        super(RotNetDataGenerator, self).__init__(N, batch_size, shuffle, None)

    def next(self):
        with self.lock:
            # get input data index and size of the current batch
            index_array, _, current_batch_size = next(self.index_generator)

        # create array to hold the images
        batch_x = np.zeros((current_batch_size,) + self.input_shape, dtype='float32')
        # create array to hold the labels
        batch_y = np.zeros(current_batch_size, dtype='float32')

        # iterate through the current batch
        for i, j in enumerate(index_array):
            image = self.images[j]

            # get a random angle
            rotation_angle = np.random.randint(360)

            # rotate the image
            rotated_image = rotate(image, rotation_angle)

            # add dimension to account for the channels if the image is greyscale
            if rotated_image.ndim == 2:
                rotated_image = np.expand_dims(rotated_image, axis=2)

            # store the image and label in their corresponding batches
            batch_x[i] = rotated_image
            batch_y[i] = rotation_angle

        # convert the numerical labels to binary labels
        batch_y = to_categorical(batch_y, 360)

        # preprocess input images
        if self.preprocess_func:
            batch_x = self.preprocess_func(batch_x)

        return batch_x, batch_y
