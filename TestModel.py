from keras.models import load_model
import sys, signal
import numpy as np
import glob
import os, time, sys
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from PIL import Image

#if ('../../PythonClient/' not in sys.path):
#    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# For safety model
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

	
# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('model/models/*.h5') 
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))


# load
safety_model = load_model(PARAMFILE)
model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')


car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))


# read a RGB image from AirSim and prepare it for consumption by the model
def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    
    return image_rgba[76:135,0:255,0:3].astype(float)

def main():
	while (True):
		# Safety model
		# Get RGBA camera images from the car
		responses = client.simGetImages([ImageRequest(1, AirSimImageType.Scene)])

		# Save it to a temporary file
		image = responses[0].image_data_uint8
		AirSimClientBase.write_file(os.path.normpath(TMPFILE), image)

		# Read-load the image as a grayscale array
		image = loadimage(TMPFILE)
			
		image = np.array(image)
		image = image.reshape(1, 256, 144, 4).astype('float32')
		
		safety = safety_model.predict(image)[0][1]
		print('safety = %.3f' % safety, end='')
			
		safety_param = safety/2.0 + 0.5	
		# Keep on the road
		car_state = client.getCarState()
		
		if (car_state.speed < 5):
			car_controls.throttle = 1.0
		else:
			car_controls.throttle = 0.0
		
		image_buf[0] = get_image()
		state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
		model_output = model.predict([image_buf, state_buf])
		car_controls.steering = round(0.5 * float(model_output[0][0]), 2) * safety_param
		
		print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))
		
		client.setCarControls(car_controls)
		
		
if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		client.enableApiControl(False)
		print('Use keyboard')