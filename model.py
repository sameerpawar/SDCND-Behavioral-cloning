import time
import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from BCNNmodel import BCNNet, LeNet

FLIP = 1
LRImages = 0

#**************************************************************************************************************
def read_data_fun(samples):
    images = [] # X_data
    angles = [] # y_data
    correction = 0.2 # this is a parameter to tune

    for line in samples:
        source_path = line[0]
        center_filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + center_filename
        center_image = cv2.imread(current_path)
        center_angle = float(line[3])
        images.append(center_image)
        angles.append(center_angle)
        if FLIP:
            image_flipped = np.fliplr(center_image)
            angle_flipped = -center_angle
            images.append(image_flipped)
            angles.append(angle_flipped)

        if LRImages:
            # append left and right images as well
            source_path = line[1]
            left_filename = source_path.split('/')[-1]
            current_path = './data/IMG/' + left_filename
            left_image = cv2.imread(current_path)
            left_angle = center_angle + correction
            images.append(left_image)
            angles.append(left_angle)            
            if FLIP:        
                image_flipped = np.fliplr(left_image)
                angle_flipped = -left_angle
                images.append(image_flipped)
                angles.append(angle_flipped)


            source_path = line[2]
            right_filename = source_path.split('/')[-1]
            current_path = './data/IMG/' + right_filename
            right_image = cv2.imread(current_path)
            right_angle = center_angle - correction
            images.append(right_image)
            angles.append(right_angle)            

            if FLIP:
                image_flipped = np.fliplr(right_image)
                angle_flipped = -right_angle
                images.append(image_flipped)
                angles.append(angle_flipped)

    X_data = np.array(images)[...,[2, 1, 0]]
    y_data = np.array(angles)
    return X_data, y_data

#**************************************************************************************************************

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('plot', 0, "The visualization flag.")
flags.DEFINE_integer('epochs', 1, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
flags.DEFINE_integer('total_samples', 10**20, "Maximum number of samples to be used for Training.")



samples = []
with open('./data/driving_log.csv') as csvfile:
    ignore_header = True
    reader = csv.reader(csvfile)
    for line in reader:
        if ignore_header:
            ignore_header = False
        else:
            samples.append(line)

        

nsamples = np.minimum(FLAGS.total_samples,len(samples))
samples = samples[:nsamples]    
X_data, y_data = read_data_fun(samples)


# Read the data
print("epochs = ", FLAGS.epochs)
print("batch size = ", FLAGS.batch_size)
print("Total samples = ", len(samples))

# Train a model
input_shape = (160, 320, 3)
model = LeNet(input_shape)
model.compile(optimizer='adam', loss='mse')
history_object = model.fit(x= X_data, y = y_data, batch_size = FLAGS.batch_size, 
epochs = FLAGS.epochs,verbose=1, validation_split=0.2, shuffle=True)
model.save('model.h5')


if FLAGS.plot > 0:
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
