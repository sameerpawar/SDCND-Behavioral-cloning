# Behavioral Cloning Neural Network model (BCNN)
from keras.layers import Input, Flatten, Dense, Conv2D, Add, Lambda, Cropping2D, MaxPooling2D
from keras.models import Model



def BCNNet(input_shape):
    # Single layer dense network
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
    conv1 = Conv2D(16, 3, padding='same', activation='relu')(norm_inp)
    fc3_inp  = Flatten()(conv1)
    out = Dense(1)(fc3_inp)
    model = Model(inp, out)
    return model


def LeNet(input_shape):
    # Single layer dense network
    inp   = Input(shape=input_shape)
    cropped_inp   = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(inp)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(cropped_inp)
    
    # convolution (5, 5, inputchannels, 6)
    conv1 = Conv2D(6, 5, padding='valid', activation='relu')(norm_inp)
    
    # max pooling with stride-2
    maxp2 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # convolution (5, 5, inputchannels, 16)
    conv3 = Conv2D(16, 5, padding='valid', activation='relu')(maxp2)

    # max pooling with stride-2
    maxp4 = MaxPooling2D(pool_size=(2, 2))(conv3)


    fc5_inp  = Flatten()(maxp4)
    fc5  = Dense(120, activation='relu')(fc5_inp)
    fc6  = Dense(84, activation='relu')(fc5)
    out = Dense(1)(fc6)
    model = Model(inp, out)
    return model


def BCNNet_nocrop(input_shape):
    # Single layer dense network
    inp   = Input(shape=input_shape)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(inp)
    conv1 = Conv2D(16, 3, padding='same', activation='relu')(norm_inp)
    fc3_inp  = Flatten()(conv1)
    out = Dense(1)(fc3_inp)
    model = Model(inp, out)
    return model

def BCNNet_3(X_data, y_data, epochs_var = 1, batch_size_var = 32, validation_split_var = 0.2):
    # Single layer dense network
    input_shape = X_data.shape[1:]
    inp   = Input(shape=input_shape)
    norm_inp = Lambda(lambda x: x/255.0 - 0.5)(inp)
    conv1 = Conv2D(16, 3, padding='same', activation='relu')(norm_inp)
    conv2 = Conv2D(16, 5, padding='same', activation='relu')(conv1)
    fc3_inp_1  = Flatten()(conv1)
    fc3_inp_2  = Flatten()(conv2)
    fc3_inp = Add()([fc3_inp_1, fc3_inp_2]) 
    out = Dense(1)(fc3_inp)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')


    # TODO: train your model here
    model.fit(X_data, y_data, 
        epochs = epochs_var, batch_size = batch_size_var, 
        validation_split = validation_split_var)
    model.save('model.h5')