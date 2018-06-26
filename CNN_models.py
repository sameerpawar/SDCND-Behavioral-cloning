# Behavioral Cloning Neural Network model (BCNN)
import tensorflow as tf
from keras.layers import Input, Flatten, Dense, Conv2D, Add, Lambda, Cropping2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, Concatenate
from keras.models import Model
 
'''
def normalized_hsv (x):
    h, s, v     = tf.split(tf.image.rgb_to_hsv(x), [1, 1, 1], 2)
    v           = tf.div(v, tf.constant(255.0))
    hsv_img     = tf.concat((h,s,v),2)
    return tf.subtract(hsv_img, tf.constant(0.5))
'''
 
def AlexNet(input_shape, BN = False, c_drop_rate = 0, fc_drop_rate = 0):
    dropout_rate = c_drop_rate + fc_drop_rate
    if dropout_rate > 0:
        DropoutFlag = True
    else:
        DropoutFlag = False

    inp             = Input(shape=input_shape)
    cropped_inp     = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp        = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
 
 
    # conv11
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv11       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    if BN:
        conv11   = BatchNormalization()(conv11)
    if DropoutFlag:
        conv11     = Dropout(c_drop_rate)(conv11)

    conv11_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv11)
 
 
 
    # conv12
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv12       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    if BN:
        conv12   = BatchNormalization()(conv12)
    if DropoutFlag:
        conv12     = Dropout(c_drop_rate)(conv12)

    conv12_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv12)
 
 
    # convolution (filters, kernel size)
    conv21       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv11_maxp)
    if BN:
        conv21   = BatchNormalization()(conv21)
    if DropoutFlag:
        conv21     = Dropout(c_drop_rate)(conv21)
    conv21_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv21)
 
 
    # convolution (filters, kernel size)
    conv22       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv12_maxp)
    if BN:
        conv22   = BatchNormalization()(conv22)
    if DropoutFlag:
        conv22     = Dropout(c_drop_rate)(conv22)
    conv22_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv22)
 
 
    conv2_merged = Concatenate(axis = -1)([conv21_maxp, conv22_maxp])
     
 
    conv31       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv2_merged)
    if BN:
        conv31   = BatchNormalization()(conv31)
    if DropoutFlag:
        conv31     = Dropout(c_drop_rate)(conv31)

    conv32       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv2_merged)
    if BN:
        conv32   = BatchNormalization()(conv32)
    if DropoutFlag:
        conv32     = Dropout(c_drop_rate)(conv32)
 
    # convolution (filters, kernel size)
    conv41       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv31)
    if BN:
        conv41   = BatchNormalization()(conv41)
    if DropoutFlag:
        conv41     = Dropout(c_drop_rate)(conv41)

    conv42       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv32)
    if BN:
        conv42   = BatchNormalization()(conv42)
    if DropoutFlag:
        conv42     = Dropout(c_drop_rate)(conv42)

    conv51       = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(conv41)
    if BN:
        conv51   = BatchNormalization()(conv51)
    if DropoutFlag:
        conv51     = Dropout(c_drop_rate)(conv51)    
    conv51_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv51)
 
    conv52       = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(conv42)
    if BN:
        conv52   = BatchNormalization()(conv52)
    if DropoutFlag:
        conv52     = Dropout(c_drop_rate)(conv52)    
    conv52_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv52)
 
    conv5_merged = Concatenate(axis = -1)([conv51_maxp, conv52_maxp])
 
     
    fc5_flat    = Flatten()(conv5_merged)
    fc5         = Dense(4096, activation='relu')(fc5_flat)
    if BN:
        fc5 = BatchNormalization()(fc5)
    if DropoutFlag:
        fc5 = Dropout(fc_drop_rate)(fc5)    
 
    fc6         = Dense(4096, activation='relu')(fc5)
    if BN:
        fc6 = BatchNormalization()(fc6)
    if DropoutFlag:
        fc6 = Dropout(fc_drop_rate)(fc6)   
 
    out         = Dense(1, activation='relu')(fc6)
 
    model       = Model(inp, out)
    return model
 

def Nvidia(input_shape, BN = False, c_drop_rate = 0, fc_drop_rate = 0):
    dropout_rate = c_drop_rate + fc_drop_rate
    if dropout_rate > 0:
        DropoutFlag = True
    else:
        DropoutFlag = False

    inp         = Input(shape=input_shape)
    cropped_inp = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    #resized_inp     = Lambda(lambda x: tf.image.resize_images(x, (200,66)))(cropped_inp)
    #norm_inp    = Lambda(lambda x: normalized_hsv(x))(resized_inp)
    norm_inp    = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)

    conv1  = Conv2D(24, 5, strides=(2, 2), padding='valid', activation='relu')(norm_inp)
    if BN:
        conv1   = BatchNormalization()(conv1)
    if DropoutFlag:
        conv1     = Dropout(c_drop_rate)(conv1)

    conv2  = Conv2D(36, 5, strides=(2, 2), padding='valid', activation='relu')(conv1)
    if BN:
        conv2   = BatchNormalization()(conv2)
    if DropoutFlag:
        conv2     = Dropout(c_drop_rate)(conv2)


    conv3  = Conv2D(48, 5, strides=(2, 2), padding='valid', activation='relu')(conv2)
    if BN:
        conv3   = BatchNormalization()(conv3)
    if DropoutFlag:
        conv3     = Dropout(c_drop_rate)(conv3)
 

    conv4  = Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu')(conv3)
    if BN:
        conv4   = BatchNormalization()(conv4)
    if DropoutFlag:
        conv4     = Dropout(c_drop_rate)(conv4)

    conv5  = Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu')(conv4)
    if BN:
        conv5   = BatchNormalization()(conv5)
    if DropoutFlag:
        conv5     = Dropout(c_drop_rate)(conv5)

    fc6_inp     = Flatten()(conv5)    
    fc6         = Dense(1164, activation='relu')(fc6_inp)
    if BN:
        fc6 = BatchNormalization()(fc6)
    if DropoutFlag:
        fc6 = Dropout(fc_drop_rate)(fc6)
    
    fc7         = Dense(100, activation='relu')(fc6)    
    if BN:
        fc7 = BatchNormalization()(fc7)
    if DropoutFlag:
        fc7 = Dropout(fc_drop_rate)(fc7)

    fc8         = Dense(50, activation='relu')(fc7)
    if BN:
        fc8 = BatchNormalization()(fc8)
    if DropoutFlag:
        fc8 = Dropout(fc_drop_rate)(fc8)
 
    fc9         = Dense(10, activation='relu')(fc8)
    if BN:
        fc9 = BatchNormalization()(fc9)
    if DropoutFlag:
        fc9 = Dropout(fc_drop_rate)(fc9)

    out  = Dense(1, activation='relu')(fc9)
    model = Model(inp, out)
    return model


def VGG_16(input_shape, BN = False, c_drop_rate = 0, fc_drop_rate = 0):
    inp         = Input(shape=input_shape)
    cropped_inp = Cropping2D(cropping=((52,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp    = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
    
    # i/p 88 x 320 x 3  
    conv1_1     = Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(norm_inp)
    conv1_2     = Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(conv1_1)
    maxpool_1   = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # i/p 44 x 160 x 64
    conv2_1     = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(maxpool_1)
    conv2_2     = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(conv2_1)
    maxpool_2   = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # i/p 22 x 80 x 128
    conv3_1     = Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(maxpool_2)
    conv3_2     = Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(conv3_1)
    conv3_3     = Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(conv3_2)
    maxpool_3   = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_3)

    # i/p 11 x 40 x 256
    conv4_1     = Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu')(maxpool_3)
    conv4_2     = Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu')(conv4_1)
    conv4_3     = Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu')(conv4_2)
    maxpool_4   = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4_3)

    # i/p 6 x 20 x 512
    conv5_1     = Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu')(maxpool_4)
    conv5_2     = Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu')(conv5_1)
    conv5_3     = Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu')(conv5_2)
    maxpool_5   = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv5_3)

    # i/p 3 x 10 x 512 = 15360
    fc1_inp     = Flatten()(maxpool_5)
    # Parameters = 15 k *1 k = 15 million
    fc1         = Dense(4096, activation='relu')(fc1_inp)
    # Parameters = 4 k *4 k = 16 million
    fc2         = Dense(4096, activation='relu')(fc1)
    out         = Dense(1, activation='relu')(fc2)
    model       = Model(inp, out)
    return model

 
def LeNet(input_shape, BN = False, c_drop_rate = 0, fc_drop_rate = 0):
    # Total parameters ~ 3.4 million
    dropout_rate = c_drop_rate + fc_drop_rate
    if dropout_rate > 0:
        DropoutFlag = True
    else:
        DropoutFlag = False

    # Input size = (160,320,3)
    # Parameters = 0
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((52, 20), (0, 0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)

    # Input size = (88,320,3)
    conv1 = Conv2D(6, 5, padding='same', activation='relu')(norm_inp) 
    if BN:
        conv1   = BatchNormalization()(conv1)
    if DropoutFlag:
        conv1     = Dropout(c_drop_rate)(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv1)

    # Input size = (44,160,6)
    conv2 = Conv2D(16, 5, padding='same', activation='relu')(maxp1)
    if BN:
        conv2 = BatchNormalization()(conv2)    
    if DropoutFlag:
        conv2 = Dropout(c_drop_rate)(conv2)        
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv2)

    # Input size = (22,80,16)
    # Parameters = 22*80*16*120 + 120 ~ 3.38 million
    fc1_inp  = Flatten()(maxp2)    
    fc1  = Dense(120, activation='relu')(fc1_inp)
    if BN:
        fc1 = BatchNormalization()(fc1)    
    if DropoutFlag:
        fc1 = Dropout(fc_drop_rate)(fc1) 

    # Input size = 120
    fc2  = Dense(84, activation='relu')(fc1)
    if BN:
        fc2 = BatchNormalization()(fc2)
    if DropoutFlag:
        fc2 = Dropout(fc_drop_rate)(fc2)

    # Input size = 84
    out = Dense(1)(fc2)
    model = Model(inp, out)
    return model 


 
def LeNet_v0(input_shape, BN = False, c_drop_rate = 0, fc_drop_rate = 0):
    # 2 conv but wider than LeNet
    dropout_rate = c_drop_rate + fc_drop_rate
    if dropout_rate > 0:
        DropoutFlag = True
    else:
        DropoutFlag = False

    # Input size = (160,320,3)
    # Parameters = 0
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((52, 20), (0, 0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)

    # Input size = (88,320,3)
    conv1 = Conv2D(16, 5, padding='valid', activation='relu')(norm_inp) 
    if BN:
        conv1   = BatchNormalization()(conv1)
    if DropoutFlag:
        conv1     = Dropout(c_drop_rate)(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv1)

    # Input size = (42,158,16)
    conv2 = Conv2D(32, 5, padding='valid', activation='relu')(maxp1)
    if BN:
        conv2 = BatchNormalization()(conv2)    
    if DropoutFlag:
        conv2 = Dropout(c_drop_rate)(conv2)        
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv2)

    # Input size = (19,77,32)
    # Parameters = 19*77*32*120 + 120 ~ 5.6 million
    fc1_inp  = Flatten()(maxp2)    
    fc1  = Dense(120, activation='relu')(fc1_inp)
    if BN:
        fc1 = BatchNormalization()(fc1)    
    if DropoutFlag:
        fc1 = Dropout(fc_drop_rate)(fc1) 

    # Input size = 120
    fc2  = Dense(84, activation='relu')(fc1)
    if BN:
        fc2 = BatchNormalization()(fc2)
    if DropoutFlag:
        fc2 = Dropout(fc_drop_rate)(fc2)

    # Input size = 84
    out = Dense(1)(fc2)
    model = Model(inp, out)
    return model 


def LeNet_v1(input_shape, BN = False, c_drop_rate = 0, fc_drop_rate = 0):
    # 3 conv but wider than LeNet
    dropout_rate = c_drop_rate + fc_drop_rate
    if dropout_rate > 0:
        DropoutFlag = True
    else:
        DropoutFlag = False

    # Input size = (160,320,3)
    # Parameters = 0
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((52, 20), (0, 0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)

    # Input size = (88,320,3)
    conv1 = Conv2D(16, 5, padding='valid', activation='relu')(norm_inp) 
    if BN:
        conv1   = BatchNormalization()(conv1)
    if DropoutFlag:
        conv1     = Dropout(c_drop_rate)(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv1)

    # Input size = (42,158,16)
    conv2 = Conv2D(32, 5, padding='valid', activation='relu')(maxp1)
    if BN:
        conv2 = BatchNormalization()(conv2)    
    if DropoutFlag:
        conv2 = Dropout(c_drop_rate)(conv2)        
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv2)

    # Input size = (19,77,32)
    conv3 = Conv2D(64, 3, padding='valid', activation='relu')(maxp2)
    if BN:
        conv3 = BatchNormalization()(conv3)    
    if DropoutFlag:
        conv3 = Dropout(c_drop_rate)(conv3)        
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv3)


    # Input size = (8,37,64)
    # Parameters = 8*37*64*120 ~ 2.3 million
    fc1_inp  = Flatten()(maxp3)    
    fc1  = Dense(120, activation='relu')(fc1_inp)
    if BN:
        fc1 = BatchNormalization()(fc1)    
    if DropoutFlag:
        fc1 = Dropout(fc_drop_rate)(fc1) 

    # Input size = 120
    fc2  = Dense(84, activation='relu')(fc1)
    if BN:
        fc2 = BatchNormalization()(fc2)
    if DropoutFlag:
        fc2 = Dropout(fc_drop_rate)(fc2)

    # Input size = 84
    out = Dense(1)(fc2)
    model = Model(inp, out)
    return model 



def LeNet_v2(input_shape, BN = False, c_drop_rate = 0, fc_drop_rate = 0):
    # 4 conv but wider than LeNet
    dropout_rate = c_drop_rate + fc_drop_rate
    if dropout_rate > 0:
        DropoutFlag = True
    else:
        DropoutFlag = False

    # Input size = (160,320,3)
    # Parameters = 0
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((52, 20), (0, 0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)

    # Input size = (88,320,3)
    conv1 = Conv2D(16, 5, padding='valid', activation='relu')(norm_inp) 
    if BN:
        conv1   = BatchNormalization()(conv1)
    if DropoutFlag:
        conv1     = Dropout(c_drop_rate)(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv1)

    # Input size = (42,158,16)
    conv2 = Conv2D(32, 5, padding='valid', activation='relu')(maxp1)
    if BN:
        conv2 = BatchNormalization()(conv2)    
    if DropoutFlag:
        conv2 = Dropout(c_drop_rate)(conv2)        
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv2)

    # Input size = (19,77,32)
    conv3 = Conv2D(64, 3, padding='valid', activation='relu')(maxp2)
    if BN:
        conv3 = BatchNormalization()(conv3)    
    if DropoutFlag:
        conv3 = Dropout(c_drop_rate)(conv3)        
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv3)


    # Input size = (8,37,64)
    conv4 = Conv2D(64, 3, padding='valid', activation='relu')(maxp3)
    if BN:
        conv4 = BatchNormalization()(conv4)    
    if DropoutFlag:
        conv4 = Dropout(c_drop_rate)(conv4)        
    maxp4 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv4)


    # Input size = (3,17,64)
    # Parameters = 3*17*64*120 ~ 0.4 million
    fc1_inp  = Flatten()(maxp4)    
    fc1  = Dense(120, activation='relu')(fc1_inp)
    if BN:
        fc1 = BatchNormalization()(fc1)    
    if DropoutFlag:
        fc1 = Dropout(fc_drop_rate)(fc1) 

    # Input size = 120
    fc2  = Dense(84, activation='relu')(fc1)
    if BN:
        fc2 = BatchNormalization()(fc2)
    if DropoutFlag:
        fc2 = Dropout(fc_drop_rate)(fc2)

    # Input size = 84
    out = Dense(1)(fc2)
    model = Model(inp, out)
    return model 


