# Collision-Avoidance-for-Self-Driving-Car
## Introduction
The first model is to make the car driving on the road without collision and the second model is to judge the safety around the car (0 represents "unsafe" and 1 represents "safe").

## File
### The fist model
- DataExplorationAndPreparation.py (including Cooking.py) is the data preprocessing for the first model.

- TrainModel.py (including Generator.py) is to train the first model.

- TestModel.py is to test the first model and make the car drive.

- TestModelsafety.py is to test the first model and simultaneously judge the safety around the car (the first and the second model)

- AirSimClient.py (to connect Airsim)

### The second model (safe)

- image_collection.py is to collect the images from Airsim.

- safe_training.py is to train the second model.

- safe_testing.py is to test the second model and judge the safety.
