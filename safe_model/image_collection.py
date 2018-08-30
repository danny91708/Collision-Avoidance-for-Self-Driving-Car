from AirSimClient import CarClient, CarControls, ImageRequest, AirSimImageType, AirSimClientBase
import pprint
import os
import time
import matplotlib.pyplot as plt

# We maintain a queue of images of this size
QUEUESIZE = 10
# Where we'll store images
IMAGEDIR = './data'

# Create image directory if it doesn't already exist
try:
    os.stat(IMAGEDIR)
except:
    os.mkdir(IMAGEDIR)
    
# connect to the AirSim simulator 
client = CarClient()
client.confirmConnection()
print('Connected')
#client.enableApiControl(True)
car_controls = CarControls()

#client.reset()

## go forward
#car_controls.throttle = 1.0
#car_controls.steering = 0
#client.setCarControls(car_controls)

cnt = 0

while True:

    # get RGBA camera images from the car
    responses = client.simGetImages([ImageRequest(1, AirSimImageType.Segmentation)])  

    image = responses[0].image_data_uint8

    AirSimClientBase.write_file(os.path.normpath(IMAGEDIR + '/image%03d.png'  % cnt ), image)
    print('save image%03d' % cnt)

    cnt += 1

    time.sleep(0.1)
