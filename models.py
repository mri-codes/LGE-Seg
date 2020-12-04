from keras.models import Model
from keras.layers import Input, Lambda, Multiply, add, concatenate, Dropout,Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import utils as kutil
import numpy as np
import os

import keras.metrics
## This Code is based on  https://github.com/rachitk/UNet-Keras-TF/blob/master/unet_build.py
#                     and https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
# Implementation heavily tweaked from https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/model/u_net.py
# to use Conv2DTranspose instead of Upsampling2D+Conv2D; also simplified Conv2D calls

gpus = '1,2,3,4'
num_gpus = 4
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

## Define dice coefficient metric and loss function associated with it
def DSC(y_true, y_pred):
    smooth = 1e-5
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_eval(y_true, y_pred):
    y_labels = y_pred.argmax(axis=-1)
    y_pred_dum =  kutil.to_categorical(y_labels, num_classes = 4)
    y_myo = y_pred_dum[:,:,:,2]
    y_scr = y_pred_dum[:,:,:,-1]
    dice_myo = DSC(y_true[:,:,:,2], y_myo)
    dice_scr = DSC(y_true[:,:,:,-1], y_scr)
    return dice_myo, dice_scr

def dice_coef_myo(y_true, y_pred):
    smooth = 1e-5
    y_myo    = K.flatten(y_pred[:,:,:,2]+y_pred[:,:,:,-1])
    true_myo = K.flatten(y_true[:,:,:,2]+y_true[:,:,:,-1])
    return (2. * K.sum(true_myo * y_myo) + smooth) / (K.sum(true_myo) + K.sum(y_myo) + smooth)

def dice_coef_scr(y_true, y_pred):
    smooth = 1e-5
    y_scr = K.flatten(y_pred[:,:,:,-1])
    true_scr = K.flatten(y_true[:,:,:,-1])
    return (2. * K.sum(true_scr * y_scr) + smooth ) / (K.sum(true_scr) + K.sum(y_scr) + smooth)

def unet_2d_shalow(input_shape=(160, 160, 2), dropout=0.25,
                 num_tissues=4, learn_rate=1e-5, loss_func='categorical_crossentropy'):
    inputs = Input(shape=input_shape)
    ChN = 32
    #160
    down1 = conv2d_block(inputs, ChN, (3, 3), activation='relu', padding='same')
    down1 = conv2d_block(down1,  ChN, (3, 3), activation='relu', padding='same')
    down1_pool = MaxPooling2D((2, 2))(down1)
    down1_pool = Dropout(dropout)(down1_pool)
    tmp = Conv2D(ChN, kernel_size=(1, 1),name='lay1_32')(inputs)
    tmp = MaxPooling2D((2, 2))(tmp)
    down1_pool = add([down1_pool,tmp])
    down1_pool = Activation('relu')(down1_pool)
    # 80
    down2 = conv2d_block(down1_pool,2*ChN, (3, 3), activation='relu', padding='same')
    down2 = conv2d_block(down2,     2*ChN, (3, 3), activation='relu', padding='same')
    down2_pool = MaxPooling2D((2, 2))(down2)
    down2_pool = Dropout(dropout)(down2_pool)#
    tmp = Conv2D( 2*ChN, kernel_size=(1, 1),name='lay2_64')(down1_pool)# match number of channels
    tmp = MaxPooling2D((2, 2))(tmp)
    down2_pool = add([down2_pool,tmp])
    down2_pool = Activation('relu')(down2_pool)
    # 40
    down3 = conv2d_block(down2_pool,4*ChN, (3, 3), activation='relu', padding='same')
    down3_pool = MaxPooling2D((2, 2))(down3)
    down3_pool = Dropout(dropout)(down3_pool)
    tmp = Conv2D(4*ChN, kernel_size=(1, 1),name='lay3_128')(down2_pool)  # match number of channels
    tmp = MaxPooling2D((2, 2))(tmp)
    down3_pool = add([down3_pool, tmp])
    down3_pool = Activation('relu')(down3_pool)
    # 20
    center = down3_pool
    # 20
    up3c = add([down3_pool, center])
    up3c = Conv2DTranspose(2*ChN, (2, 2), strides=(2, 2), padding='same')(up3c)
    up3 = conv2d_block(up3c,2*ChN, (2, 2), activation='relu', padding='same')
    up3 = Dropout(dropout)(up3)
    tmp = Conv2D(2*ChN, kernel_size=(1, 1),name='lay4_64')(up3c)  # match number of channels
    up3c = add([up3, tmp])
    up3c = Activation('relu')(up3c)
    # 40
    up2 = add([down2_pool, up3c])
    up2 = Conv2DTranspose(ChN, (3, 3), strides=(2, 2), padding='same')(up2)
    up2c = conv2d_block(up2,ChN, (3, 3), activation='relu', padding='same')
    up2c = Dropout(dropout)(up2c)
    tmp = Conv2D(ChN, kernel_size=(1, 1),name='lay5_32')(up2)  # match number of channels
    up2c = add([up2c, tmp])
    up2c = Activation('relu')(up2c)
    # 80
    up1 = add([down1_pool, up2c])
    up1 = Conv2DTranspose(ChN, (2, 2), strides=(2, 2), padding='same')(up1)
    up1c = conv2d_block(up1,ChN, (3, 3), activation='relu', padding='same')
    up1c = conv2d_block(up1c, ChN, (3, 3), activation='relu', padding='same')
    up1c = Dropout(dropout)(up1c)
    tmp = Conv2D(ChN, kernel_size=(1, 1),name='lay6_16')(up1)  # match number of channels
    up1c = add([up1c, tmp])
    up1c = Activation('relu')(up1c)
    # 160
    high_res_skiplong = add([up1c, down1])
    high_res_skiplong = conv2d_block(high_res_skiplong, ChN, (3, 3), activation='relu', padding='same')  #
    high_res_skiplong = conv2d_block(high_res_skiplong, ChN, (3, 3), activation='relu', padding='same')  #

    classify = conv2d_block(high_res_skiplong,num_tissues, (1, 1), activation='relu')
    classify = Activation('softmax')(classify)

    model = Model(inputs=inputs, outputs=classify, name='my_umodel')
    model.summary()
    parallel_model = kutil.multi_gpu_model(model, num_gpus)
    parallel_model.compile(optimizer=Adam(lr=learn_rate), loss=loss_func, metrics=[dice_coef_myo, dice_coef_scr])
    parallel_model.summary()

    return parallel_model


def unet_2d_shalow_baseline(input_shape=(160, 160, 2), dropout=0.25,
                 num_tissues=4, learn_rate=1e-5, loss_func='categorical_crossentropy'):
    inputs = Input(shape=input_shape) # 2D image; 3 channels: 2Cine+1LGE
    ChN = 48
    #160
    down1 = conv2d_block(inputs, ChN, (3, 3), activation='relu', padding='same') # 4*ChN
    down1 = conv2d_block(down1,  ChN, (3, 3), activation='relu', padding='same')
    down1_pool = MaxPooling2D((2, 2))(down1)
    down1_pool = Dropout(dropout)(down1_pool) #
    tmp = Conv2D(ChN, kernel_size=(1, 1),name='lay1_32')(inputs)
    tmp = MaxPooling2D((2, 2))(tmp)
    down1_pool = add([down1_pool,tmp])
    down1_pool = Activation('relu')(down1_pool)
    # 80
    down2 = conv2d_block(down1_pool,2*ChN, (3, 3), activation='relu', padding='same')
    down2 = conv2d_block(down2,     2*ChN, (3, 3), activation='relu', padding='same')
    down2_pool = MaxPooling2D((2, 2))(down2)
    down2_pool = Dropout(dropout)(down2_pool)
    tmp = Conv2D( 2*ChN, kernel_size=(1, 1),name='lay2_64')(down1_pool)# match number of channels
    tmp = MaxPooling2D((2, 2))(tmp)
    down2_pool = add([down2_pool,tmp])
    down2_pool = Activation('relu')(down2_pool)
    # 40
    down3 = conv2d_block(down2_pool,4*ChN, (3, 3), activation='relu', padding='same')

    down3_pool = MaxPooling2D((2, 2))(down3)
    down3_pool = Dropout(dropout)(down3_pool)
    tmp = Conv2D(4*ChN, kernel_size=(1, 1),name='lay3_128')(down2_pool)  # match number of channels
    tmp = MaxPooling2D((2, 2))(tmp)
    down3_pool = add([down3_pool, tmp])
    down3_pool = Activation('relu')(down3_pool)
    # 20
    center = down3_pool # no convolution applied here to limit network size
    # 20
    up3c = add([down3_pool, center])
    up3c = Conv2DTranspose(2*ChN, (2, 2), strides=(2, 2), padding='same')(up3c)
    up3 = conv2d_block(up3c,2*ChN, (2, 2), activation='relu', padding='same')
    up3 = Dropout(dropout)(up3)
    tmp = Conv2D(2*ChN, kernel_size=(1, 1),name='lay4_64')(up3c)  # match number of channels
    up3c = add([up3, tmp])
    up3c = Activation('relu')(up3c)
    # 40
    up2 = add([down2_pool, up3c])
    up2 = Conv2DTranspose(ChN, (3, 3), strides=(2, 2), padding='same')(up2)
    up2c = conv2d_block(up2,ChN, (3, 3), activation='relu', padding='same')
    up2c = Dropout(dropout)(up2c)
    tmp = Conv2D(ChN, kernel_size=(1, 1),name='lay5_32')(up2)  # match number of channels
    up2c = add([up2c, tmp])
    up2c = Activation('relu')(up2c)
    # 80  # 2,2
    up1 = add([down1_pool, up2c])
    up1 = Conv2DTranspose(ChN, (2, 2), strides=(2, 2), padding='same')(up1)
    up1c = conv2d_block(up1,ChN, (3, 3), activation='relu', padding='same')
    up1c = conv2d_block(up1c, ChN, (3, 3), activation='relu', padding='same')
    up1c = Dropout(dropout)(up1c)
    tmp = Conv2D(ChN, kernel_size=(1, 1),name='lay6_16')(up1)  # match number of channels
    up1c = add([up1c, tmp])
    up1c = Activation('relu')(up1c)
    # 160
    high_res_skiplong = add([up1c, down1])
    high_res_skiplong = conv2d_block(high_res_skiplong, ChN, (3, 3), activation='relu', padding='same')  # extras
    high_res_skiplong = conv2d_block(high_res_skiplong, ChN, (3, 3), activation='relu', padding='same')  # extras

    classify = conv2d_block(high_res_skiplong,num_tissues, (1, 1), activation='relu')
    classify = Activation('softmax')(classify)

    model = Model(inputs=inputs, outputs=classify, name='my_umodel')

    model.summary()
    parallel_model = kutil.multi_gpu_model(model, num_gpus)
    parallel_model.compile(optimizer=Adam(lr=learn_rate), loss=loss_func, metrics=[dice_coef_myo, dice_coef_scr])
    parallel_model.summary()

    return parallel_model


def conv2d_block(input_tensor, n_filters, kernel_size=(3, 3), activation='relu', padding='same', batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding=padding)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal",
               padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x