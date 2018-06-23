# Behavioral Cloning Neural Network model (BCNN)
from keras.layers import Input, Flatten, Dense, Conv2D, Add, Lambda, Cropping2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, Concatenate
from keras.models import Model


def LeNet(input_shape):
    # Vanilla LeNet
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
    
    # convolution (5, 5, inputchannels, 6)
    conv1 = Conv2D(6, 5, padding='valid', activation='relu')(norm_inp)    
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv1)

    # convolution (5, 5, inputchannels, 16)
    conv2 = Conv2D(16, 5, padding='valid', activation='relu')(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv2)

    fc1_inp  = Flatten()(maxp2)    
    fc1  = Dense(120, activation='relu')(fc1_inp)

    fc2  = Dense(84, activation='relu')(fc1)
    
    out = Dense(1)(fc2)
    model = Model(inp, out)
    return model

def LeNet_D(input_shape):
    # Vanilla LeNet + Dropout
    conv_drop_rate = 0.1
    fc_drop_rate = 0.2
    
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
    
    # convolution (5, 5, inputchannels, 6)
    conv1       = Conv2D(6, 5, padding='valid', activation='relu')(norm_inp)
    conv1_out   = Dropout(conv_drop_rate)(conv1)    
    maxp1       = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv1_out)


    conv2       = Conv2D(16, 5, padding='valid', activation='relu')(maxp1)
    conv2_out   = Dropout(conv_drop_rate)(conv2)    
    maxp2       = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv2_out)


    fc1_inp     = Flatten()(maxp2)    
    fc1         = Dense(120, activation='relu')(fc1_inp)
    fc1_out     = Dropout(fc_drop_rate)(fc1)


    fc2         = Dense(84, activation='relu')(fc1_out)
    fc2_out     = Dropout(fc_drop_rate)(fc2)

    out         = Dense(1)(fc2_out)
    model       = Model(inp, out)
    return model

def LeNet_BN(input_shape):
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
    
    # convolution (5, 5, inputchannels, 6)
    conv1 = Conv2D(6, 5, padding='valid', activation='relu')(norm_inp)    
    conv1_norm = BatchNormalization()(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv1_norm)

    # convolution (5, 5, inputchannels, 16)
    conv2 = Conv2D(16, 5, padding='valid', activation='relu')(maxp1)    
    conv2_norm = BatchNormalization()(conv2)
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2))(conv2_norm)

    
    fc1_inp  = Flatten()(maxp2)    
    fc1  = Dense(120, activation='relu')(fc1_inp)
    fc1_norm = BatchNormalization()(fc1)

    fc2  = Dense(84, activation='relu')(fc1_norm)
    fc2_norm = BatchNormalization()(fc2)
    
    out = Dense(1)(fc2_norm)
    model = Model(inp, out)
    return model

def Nvidia_D(input_shape):
    # Nvidia + Dropout
    conv_drop_rate = 0
    fc_drop_rate   = 0

    BN = 0
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
    
    # convolution (5, 5, inputchannels, 24)
    conv1      = Conv2D(24, 5, padding='same')(norm_inp)
    if BN:
        conv1 = BatchNormalization()(conv1)
    
    conv1  = Activation('relu')(conv1)
    # max pooling with stride-2
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1_out  = Dropout(conv_drop_rate)(conv1)

    # convolution (5, 5, inputchannels, 36)
    conv2      = Conv2D(36, 5, padding='same')(conv1_out)
    if BN:
        conv2 = BatchNormalization()(conv2)
    
    conv2  = Activation('relu')(conv2)
    # max pooling with stride-2
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2_out  = Dropout(conv_drop_rate)(conv2)


    # convolution (5, 5, inputchannels, 48)
    conv3      = Conv2D(24, 5, padding='same')(conv2_out)
    if BN:
        conv3 = BatchNormalization()(conv3)
    
    conv3  = Activation('relu')(conv3)
    # max pooling with stride-2
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3_out  = Dropout(conv_drop_rate)(conv3)



    # convolution (3, 3, inputchannels, 64)
    conv4      = Conv2D(64, 3, padding='same')(conv3_out)
    if BN:
        conv4 = BatchNormalization()(conv4)
    
    conv4  = Activation('relu')(conv4)
    # max pooling with stride-2
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out  = Dropout(conv_drop_rate)(conv4)


    # convolution (5, 5, inputchannels, 24)
    conv5      = Conv2D(64, 3, padding='same')(conv4_out)
    if BN:
        conv5 = BatchNormalization()(conv5)
    
    conv5  = Activation('relu')(conv5)
    # max pooling with stride-2
    conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv5_out  = Dropout(conv_drop_rate)(conv5)

    fc6_inp  = Flatten()(conv5_out)    
    fc6  = Dense(100, activation='relu')(fc6_inp)
    fc6_out  = Dropout(fc_drop_rate)(fc6)

    fc7  = Dense(50, activation='relu')(fc6_out)
    fc7_out  = Dropout(fc_drop_rate)(fc7)


    fc8  = Dense(10, activation='relu')(fc7_out)
    fc8_out  = Dropout(fc_drop_rate)(fc8)

    out  = Dense(1, activation='relu')(fc8_out)
    model = Model(inp, out)
    return model

def AlexNet(input_shape):
    inp             = Input(shape=input_shape)
    cropped_inp     = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp        = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)


    # conv11
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv11       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    conv11_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv11)



    # conv12
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv12       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    conv12_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv12)


    # convolution (filters, kernel size)
    conv21       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv11_maxp)
    conv21_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv21)


    # convolution (filters, kernel size)
    conv22       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv12_maxp)
    conv22_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv22)


    conv2_merged = Concatenate(axis = -1)([conv21_maxp, conv22_maxp])
    

    conv31       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv2_merged)
    conv32       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv2_merged)

    # convolution (filters, kernel size)
    conv41       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv31)
    conv42       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv32)
    
    conv51       = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(conv41)
    conv51_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv51)

    conv52       = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(conv42)
    conv52_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv52)

    conv5_merged = Concatenate(axis = -1)([conv51_maxp, conv52_maxp])

    
    fc5_inp     = Flatten()(conv5_merged)
    fc5_out     = Dense(4096, activation='relu')(fc5_inp)

    fc6_out     = Dense(4096, activation='relu')(fc5_out)

    out         = Dense(1, activation='relu')(fc6_out)

    model       = Model(inp, out)
    return model

def AlexNet_v2(input_shape):
    inp             = Input(shape=input_shape)
    cropped_inp     = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp        = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)


    # conv11
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv11       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    conv11_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv11)



    # conv12
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv12       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    conv12_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv12)


    # convolution (filters, kernel size)
    conv21       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv11_maxp)
    conv21_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv21)


    # convolution (filters, kernel size)
    conv22       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv12_maxp)
    conv22_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv22)


    conv2_merged = Concatenate(axis = -1)([conv21_maxp, conv22_maxp])
    

    conv3        = Conv2D(384, 3, strides=(1, 1), padding='same', activation='relu')(conv2_merged)
    conv3_maxp   = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv3)

    
    fc5_inp     = Flatten()(conv3_maxp)
    fc5_out     = Dense(2048, activation='relu')(fc5_inp)

    fc6_out     = Dense(200, activation='relu')(fc5_out)

    out         = Dense(1, activation='relu')(fc6_out)

    model       = Model(inp, out)
    return model    


def AlexNet_v1(input_shape):
    inp             = Input(shape=input_shape)
    cropped_inp     = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp        = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)


    # conv11
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv11       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    conv11_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv11)



    # conv12
    # conv(11, 11, 48, 4, 4, padding='VALID', name='conv1')
    conv12       = Conv2D(48, 11, strides=(4, 4), padding='same', activation='relu')(norm_inp)
    conv12_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv12)


    # convolution (filters, kernel size)
    conv21       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv11_maxp)
    conv21_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv21)


    # convolution (filters, kernel size)
    conv22       = Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu')(conv12_maxp)
    conv22_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv22)


    conv2_merged = Concatenate(axis = -1)([conv21_maxp, conv22_maxp])
    

    conv31       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv2_merged)
    conv32       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv2_merged)

    # convolution (filters, kernel size)
    conv41       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv31)
    conv42       = Conv2D(192, 3, strides=(1, 1), padding='same', activation='relu')(conv32)
    
    conv51       = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(conv41)
    conv51_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv51)

    conv52       = Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(conv42)
    conv52_maxp  = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv52)

    conv5_merged = Concatenate(axis = -1)([conv51_maxp, conv52_maxp])

    
    fc5_inp     = Flatten()(conv5_merged)
    fc5_out     = Dense(4096, activation='relu')(fc5_inp)

    fc6_out     = Dense(4096, activation='relu')(fc5_out)

    out         = Dense(1, activation='relu')(fc6_out)

    model       = Model(inp, out)
    return model


#Read on saved webpage about AlexNet finetuning.

'''
ResNet
VGG
'''
