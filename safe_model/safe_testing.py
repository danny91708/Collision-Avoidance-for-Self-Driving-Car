from AirSimClient import CarClient, CarControls, ImageRequest, AirSimImageType, AirSimClientBase
import os
import time
import tensorflow as tf
import pickle
import sys
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from PIL import Image
import numpy as np

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Where we've stored images
IMAGEDIR = './test_data'

TMPFILE = IMAGEDIR + '/active.png'
PARAMFILE = 'safety_model2.h5'

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


# connect to the AirSim simulator 
client = CarClient()
client.confirmConnection()
print('Connected')
#client.enableApiControl(True)
car_controls = CarControls()

#client.reset()

# go forward
#car_controls.throttle = INITIAL_THROTTLE
#car_controls.steering = 0
#client.setCarControls(car_controls)

# load
model = load_model(PARAMFILE)

while True:
    # Get RGBA camera images from the car
    responses = client.simGetImages([ImageRequest(1, AirSimImageType.Scene)])

    # Save it to a temporary file
    image = responses[0].image_data_uint8
    AirSimClientBase.write_file(os.path.normpath(TMPFILE), image)

    # Read-load the image as a grayscale array
    image = loadimage(TMPFILE)
        
    image = np.array(image)
    image = image.reshape(1, 256, 144, 4).astype('float32')
    
    print('safety = %.3f' % model.predict(image)[0][1])

    # Wait a bit on each iteration
    time.sleep(0.1)