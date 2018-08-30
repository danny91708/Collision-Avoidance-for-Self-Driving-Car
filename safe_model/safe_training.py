# Built-in modules
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import h5py
from PIL import Image

# Final image is crash; previous are no-crash
SAFESIZE = 5

# Images are too big to train quickly, so we scale 'em down
SCALEDOWN = 6

# Where we've stored images
SAFEIMAGEDIR = './data/safe_image'
UNSAFEIMAGEDIR = './data/unsafe_image'

# Where we'll store weights and biases
PARAMFILE = 'safety_model.h5'


def loadimage(filename):
    '''
    Loads an RGBA image from FILENAME, converts it to grayscale, and returns a flattened copy
    '''
    pic = Image.open(filename)
    pic_rz = pic.resize((256,144), Image.ANTIALIAS)
    np_pic= np.array(pic_rz,float)
    np_pic = np_pic.ravel()
    np_pic /= 255

    return np_pic

def CNN_model():
    model = Sequential()
    model.add(Conv2D(filters=16,
                     kernel_size=(5, 5),
                     padding='same',
                     input_shape=(256, 144, 4),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=36,
                     kernel_size=(5, 5),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=36,
                     kernel_size=(5, 5),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=36,
                     kernel_size=(5, 5),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


def main():

    # This will get number of pixels in each image (they must all be the same!)
    imgsize = 0

    # Read safe image from car, convert to grayscale, scale down, and flatten for use as input
    images = []
    print('Data preprocessing ...')
    for filename_safe in os.listdir(SAFEIMAGEDIR):
        #print('Loading safe image: %s' % filename_safe)
        image = loadimage(SAFEIMAGEDIR + '/' + filename_safe)
        images.append(image)

    # Size of the safe images
    SAFESIZE = len(images)

    # Read unsafe image from car, convert to grayscale, scale down, and flatten for use as input
    for filename_unsafe in os.listdir(UNSAFEIMAGEDIR):
        #print('Loading unsafe image: %s' % filename_unsafe)
        image = loadimage(UNSAFEIMAGEDIR + '/' + filename_unsafe)
        imgsize = np.prod(image.shape)
        images.append(image)

    images = np.array(images)
    images = images.reshape(images.shape[0], 256, 144, 4).astype('float32')

    print('Label targets ...')
    # 01 = safe, 10 = unsafe
    targets = []
    for k in range(len(images)):
        if k < SAFESIZE:
            targets.append([0,1])
        else:
            targets.append([1,0])
    targets = np.array(targets)

    # Training
    print('Start training ...')
    model = CNN_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=images, y=targets, epochs=100, batch_size=32, verbose=2)

    # Save
    print('Save model: ', model.predict(images))
    model.save(PARAMFILE)

if __name__ == '__main__':

    main()