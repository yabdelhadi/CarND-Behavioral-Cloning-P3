import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

samples = []
# read in the image location and file name from the excel sheet
with open('Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2 # Adding a tuning parameter for steering correction for images captured by left and right camera
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # read in images from center, left and right cameras
                center_image = cv2.imread(batch_sample[0].strip())
                left_image = cv2.imread(batch_sample[1].strip())
                right_image = cv2.imread(batch_sample[2].strip())
                # create adjusted steering measurements for the side camera images
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # add images and angles to data set
                images.extend([center_image, left_image, right_image])
                angles.extend([steering_center, steering_left, steering_right])
                
            augmented_images = []
            augmented_angles = []
            # augmented the data by filpping the images
            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

keep_prob = 0.5

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(keep_prob))
model.add(Dense(50))
model.add(Dropout(keep_prob))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch = \
                    len(train_samples), validation_data = validation_generator, \
                    nb_val_samples = len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5') 

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
