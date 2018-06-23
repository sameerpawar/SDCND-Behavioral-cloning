from keras.layers import Input, Flatten, Dense, Conv2D, Lambda, Cropping2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout
from keras.models import Model

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

    out         = Dense(1000, activation='relu')(fc6_out)

    model       = Model(inp, out)
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

    out         = Dense(1000, activation='relu')(fc6_out)

    model       = Model(inp, out)
    return model    