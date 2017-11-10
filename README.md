# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_2017_10_27_21_24_41_475.jpg "Driving in the center"
[image3]: ./examples/center_2017_10_27_21_25_23_991.jpg "Recovery Image"
[image4]: ./examples/center_2017_10_27_21_25_44_078.jpg "Recovery Image"
[image5]: ./examples/center_2017_10_27_21_25_53_685.jpg "Recovery Image"
[image6]: ./examples/center_2017_10_27_21_25_56_687.jpg "Normal Image"
[image7]: ./examples/center_2017_10_27_21_25_56_687_flipped.jpg "Flipped Image"
[image8]: ./examples/loss.png "Loss Figure"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 convolution layers with depths of 24, 36, and 48 (model.py lines 74-76) and two 3x3 convolution layers with depths of 64 (model.py lines 77-78) 

The model includes RELU layers to introduce nonlinearity (model.py lines 74-78), and the data is normalized in the model using a Keras lambda layer (model.py line 72). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 81 & 83). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 89-91). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving counter-clockwise 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model architecture and add more layers as needed.

My first step was to use a convolution neural network model similar to the NVIDIA End-to-End Deep Learning Network I thought this model might be appropriate because I was developed for the same purpose which is cloning human behavior using three cameras. The NVIDIA End-to-End Deep Learning Network consists of 9 layers

![alt text][image1]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set, but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I normalized all the data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I added two dropout layers.

The vehicle was driving a lot better around the track, but the model was struggling with the left and right curve right after the bridge sine those curves are very sharp. I collected more data for those two curves to teach a model how to recover when the vehicle starts to go off the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 72-85) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         			|     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         			| 160x320x3 image   							| 
| Normalization  			| Normalizing the image using Keras Lambda layer 					|
| Cropping  			| Cropping the image to crop the area of interest using the keras Cropping2D layer   					|
| Convolution 5x5     		| 5x5 kernel, 2x2 strides, 24 filters 	|
| RELU						| Activation Function							|
| Convolution 5x5	    	| 5x5 kernel, 2x2 strides, 36 filters	|
| RELU						| Activation Function							|
| Convolution 5x5	    	| 5x5 kernel, 2x2 strides, 48 filters	|
| RELU						| Activation Function							|
| Convolution 5x5	    	| 3x3 kernel, 2x2 strides, 64 filters	|
| RELU						| Activation Function							|
| Convolution 5x5	    	| 3x3 kernel, 2x2 strides, 64 filters	|
| RELU						| Activation Function							|
| Flatten		      		| 			    |
| Fully connected			| 100      				|
| Dropout					| 50% keep probability							|
| Fully connected			| 50   			    	|
| Dropout					| 50% keep probability							|
| Fully connected			| 10   			    	|
| Fully connected			| 1   			    	|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to increase the steering angle when the vehicle is leaving the road to recover back to the center. An examples of these recovery images are shown below:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would eliminate the model from overfitting. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 47,430 number of data points. I then preprocessed this data by reading in each image and flipping it. I also used all three cameras center, left, and right but I add 0.2 deg of steering angle to left camera image and subtracted 0.2 deg of steering angle to right camera image

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by training and validation loss shown in the figure below

![alt text][image8]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Overall the model work very well however I tried to go beyond the scope of this project and higher the vehicle speed of the simulator, but the vehicle wasn't tracking as well on the center of the road as at low speed. I believe this can be solved by collecting more data or by making the model smarter by predicting different steering angle based on vehicle speed.
