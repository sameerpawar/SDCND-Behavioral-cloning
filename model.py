import time
import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from CNN_models import LeNet, LeNet_v0, LeNet_v1, LeNet_v2
from CNN_models import Nvidia


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


#**************************************************************************************************************
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2 # this is a parameter to tune
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)            

                if FLIP:
                    # Flip the center image
                    image_flipped = np.fliplr(center_image)
                    angle_flipped = -center_angle
                    images.append(image_flipped)
                    angles.append(angle_flipped)

                if LRImages:
                    # append left and right images as well
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name)
                    left_angle = center_angle + correction
                    images.append(left_image)
                    angles.append(left_angle)            
                    if FLIP:        
                        # Flip the left image
                        image_flipped = np.fliplr(left_image)
                        angle_flipped = -left_angle
                        images.append(image_flipped)
                        angles.append(angle_flipped)

                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name)
                    right_angle = center_angle - correction
                    images.append(right_image)
                    angles.append(right_angle)            
                    if FLIP:
                        # Flip the right image        
                        image_flipped = np.fliplr(right_image)
                        angle_flipped = -right_angle
                        images.append(image_flipped)
                        angles.append(angle_flipped)
            X_train = np.array(images)[...,[2, 1, 0]]
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

'''
#**************************************************************************************************************



flags = tf.app.flags
FLAGS = flags.FLAGS
# command line flags
flags.DEFINE_string('model', 'LeNet', "Model selection.")
flags.DEFINE_string('data', 'data', "training data folder.")
flags.DEFINE_string('out', 'model.h5', "Output file name.")
flags.DEFINE_integer('start_line', 0, "Starting of the line from excel.")
flags.DEFINE_integer('end_line', 10**20, "Ending of the line from excel.")
flags.DEFINE_integer('plot', 0, "The visualization flag.")
flags.DEFINE_boolean('flip', False, "Flip images boolean.")
flags.DEFINE_boolean('LR', False, "Add left-right images boolean.")
flags.DEFINE_integer('epochs', 1, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
flags.DEFINE_boolean('BN', False, "batch normalization.")
flags.DEFINE_float('conv_drop', 0.0, "Drop rate for conv layers.")
flags.DEFINE_float('fc_drop', 0.0, "Drop rate for fully connected layers.")
flags.DEFINE_float('rate', 1e-3, "Learning rate for fully connected layers.")

FLIP = FLAGS.flip
LRImages = FLAGS.LR

input_shape = (160, 320, 3)

switcher = {
        'LeNet':        LeNet(input_shape, BN = FLAGS.BN,  c_drop_rate = FLAGS.conv_drop, fc_drop_rate = FLAGS.fc_drop),   
        'LeNet_v0':     LeNet_v0(input_shape, BN = FLAGS.BN,  c_drop_rate = FLAGS.conv_drop, fc_drop_rate = FLAGS.fc_drop),   
        'LeNet_v1':     LeNet_v1(input_shape, BN = FLAGS.BN,  c_drop_rate = FLAGS.conv_drop, fc_drop_rate = FLAGS.fc_drop),   
        'LeNet_v2':     LeNet_v2(input_shape, BN = FLAGS.BN,  c_drop_rate = FLAGS.conv_drop, fc_drop_rate = FLAGS.fc_drop),
        'Nvidia':       Nvidia(input_shape, BN = FLAGS.BN,  c_drop_rate = FLAGS.conv_drop, fc_drop_rate = FLAGS.fc_drop)
    }

samples = []
start_line = FLAGS.start_line
end_line = FLAGS.end_line

with open('./data/driving_log.csv') as csvfile:
    line_counter = 0
    reader = csv.reader(csvfile)
    for line in reader:
        if line_counter > start_line and line_counter < end_line:
            samples.append(line)
        line_counter += 1


# Train a model
model = switcher.get(FLAGS.model, "Invalid model")
optm = Adam(lr=FLAGS.rate)
#optm  = SGD(lr=FLAGS.rate, momentum=0.9, decay=0.0, nesterov=True)
model.compile(optimizer=optm, loss='mse')



#**************************************************************************************************************
# If no generator
X_data, y_data = read_data_fun(samples, FLAGS.data)
history_object = model.fit(x= X_data, y = y_data, batch_size = FLAGS.batch_size, 
epochs = FLAGS.epochs,verbose=1, validation_split=0.2, shuffle=True)
#**************************************************************************************************************


#**************************************************************************************************************
'''
# If generator
train_generator         = generator(train_samples, FLAGS.batch_size)
validation_generator    = generator(validation_samples, FLAGS.batch_size)
# Train a model
history_object = model.fit_generator(train_generator, steps_per_epoch =
len(train_samples), validation_data = 
validation_generator,
validation_steps = len(validation_samples), 
epochs=FLAGS.epochs, verbose=1)
'''
#**************************************************************************************************************


model.save(FLAGS.out)
print("epochs = ", FLAGS.epochs)
print("batch size = ", FLAGS.batch_size)
print("Using model " + FLAGS.model)
print("Output filename = " + FLAGS.out)
if FLIP:
    print("Flip enabled")
if FLAGS.BN:
    print("BN enabled")
if LRImages:
    print("left-right images used")

if FLAGS.conv_drop + FLAGS.fc_drop > 0:
    print("Drop out used")

print("Learning rate used = ", FLAGS.rate)
    
if FLAGS.plot > 0:
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
