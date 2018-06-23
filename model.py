import time
import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from BCNNmodel import LeNet, LeNet_D, LeNet_BN
from BCNNmodel import AlexNet, AlexNet_v1
#from BCNNmodel import Nvidia, Nvidia_D, Nvidia_BN
#from BCNNmodel import ResNet

#Course data 0 - 8036
# two laps of track 1: 8037-12618
# two laps of recovery data track-1: 

#**************************************************************************************************************
def read_data_fun(samples, src):
    images = [] # X_data
    angles = [] # y_data
    correction = 0.2 # this is a parameter to tune

    for line in samples:
        source_path = line[0]
        center_filename = source_path.split('/')[-1]
        current_path = './' + src + '/IMG/' + center_filename
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
            current_path = './' + src + '/IMG/' + left_filename
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
            current_path = './' + src + '/IMG/' + right_filename
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
flags.DEFINE_string('model', 'LeNet', "Model selection.")
flags.DEFINE_string('data', 'data', "training data folder.")
flags.DEFINE_integer('start_line', 0, "Starting of the line from excel.")
flags.DEFINE_integer('end_line', 10**20, "Ending of the line from excel.")
flags.DEFINE_integer('plot', 0, "The visualization flag.")
flags.DEFINE_boolean('flip', False, "Flip images boolean.")
flags.DEFINE_boolean('LR', False, "Add left-right images boolean.")
flags.DEFINE_integer('epochs', 1, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
flags.DEFINE_integer('total_samples', 10**20, "Maximum number of samples to be used for Training.")

FLIP = FLAGS.flip
LRImages = FLAGS.LR

input_shape = (160, 320, 3)

switcher = {
        'LeNet':         LeNet(input_shape),   
        'LeNet_BN':      LeNet_BN(input_shape),
        'LeNet_D':       LeNet_D(input_shape),
        'AlexNet':       AlexNet(input_shape),
        'AlexNet_v1':    AlexNet_v1(input_shape)
    }


samples = []
start_line = FLAGS.start_line
end_line = FLAGS.end_line

with open('./data/driving_log.csv') as csvfile:
    ignore_header = True
    line_counter = 0
    reader = csv.reader(csvfile)
    for line in reader:
        if line_counter > start_line and line_counter < end_line:
            samples.append(line)
        line_counter += 1

        

nsamples = np.minimum(FLAGS.total_samples,len(samples))
samples = samples[:nsamples]    
X_data, y_data = read_data_fun(samples, FLAGS.data)


# Read the data
print("epochs = ", FLAGS.epochs)
print("batch size = ", FLAGS.batch_size)





# Train a model

#argument = 'Nvidia_D_v2'
print("Using model " + FLAGS.model)
print()
model = switcher.get(FLAGS.model, "Invalid model")
#model = LeNet(input_shape)
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
