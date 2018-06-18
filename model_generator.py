import time
import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from BCNNmodel import BCNNet


#**************************************************************************************************************
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                # TODO append left and right images as well
                images.append(center_image)
                angles.append(center_angle)            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#**************************************************************************************************************

def read_data_fun(nsamples = 1e12):
    lines = []
    samples = 0
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        ignore_header = True
        for line in reader:
            if ignore_header or samples > nsamples:
                ignore_header = False
            else:
                lines.append(line)
                samples += 1


    images = [] # X_data
    measurements = [] # Y_data

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        #image = ((image) - 128.0)/128.0
        measurement = float(line[3])
        images.append(image)
        measurements.append(measurement)

    X_data = np.array(images)[...,[2, 1, 0]]
    y_data = np.array(measurements)

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
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Read the data
print("epochs = ", FLAGS.epochs)
print("batch size = ", FLAGS.batch_size)
print("Training samples = ", len(train_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, FLAGS.batch_size)
validation_generator = generator(validation_samples, FLAGS.batch_size)

# Train a model
input_shape = (160, 320, 3)
model = BCNNet(input_shape)
model.compile(optimizer='adam', loss='mse')
history_object = model.fit_generator(train_generator, steps_per_epoch =
len(train_samples), validation_data = 
validation_generator,
validation_steps = len(validation_samples), 
epochs=FLAGS.epochs, verbose=1)
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
