#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:10:12 2018

@author: yzj
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import keras as K
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import concatenate
from keras.layers.convolutional import Convolution2D, ZeroPadding2D,MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.regularizers import l1_l2
import random
import numpy as np
import scipy.io as sio  
import sys
import pickle as pickle
import gzip
import matplotlib.pyplot as plt


def dmlp(Xt=None,Yt=None, weights=None, numofc=2, input_tensor=None, 
                   include_top=True):

    numofdim = Xt.shape[2]
    regpara = 0.0000
    dopara = 0.5
    
    input_img = Input(shape=(1,numofdim,1))  # adapt this if using `channels_first` image data format

#    x = Convolution2D(128,1,numofdim, border_mode='valid')(input_img)
#    x = Activation('sigmoid')(x)
##    x = Convolution2D(256,1,1, border_mode='same')(x)
##    x = Activation('sigmoid')(x)
#    x = Convolution2D(512,1,1, border_mode='same')(x)
#    x = Activation('sigmoid')(x)
    x = Flatten()(input_img)
    x = Dense(512, activation='tanh', name='hidden0', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    x = Dropout(dopara)(x)
    x = Dense(512, activation='tanh', name='hidden1', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    x = Dropout(dopara)(x)
    x = Dense(512, activation='tanh', name='hidden2', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    x = Dropout(dopara)(x)
    x = Dense(1024, activation='sigmoid', name='hidden3', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    x = Dropout(dopara)(x)
    out = Dense(numofc, activation='softmax', name='output')(x)
    
    model = Model(input_img,out)
    model.summary()
    return model


def dense_mlp(Xt=None,Yt=None, weights=None, numofc=2, input_tensor=None, 
                   include_top=True):

    numofdim = Xt.shape[2]
    regpara = 0.0000
    dopara = 0.5
    
    input_img = Input(shape=(1,numofdim,1))  # adapt this if using `channels_first` image data format

    x = Flatten()(input_img)
    y = Dense(512, activation='tanh', name='hidden0', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    y = Dropout(dopara)(y)
    x = concatenate([x,y],axis=1)
    y = Dense(512, activation='tanh', name='hidden1', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    y = Dropout(dopara)(y)
    x = concatenate([x,y],axis=1)
    y = Dense(512, activation='tanh', name='hidden2', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    y = Dropout(dopara)(y)
    x = concatenate([x,y],axis=1)
    x = Dense(1024, activation='sigmoid', name='hidden3', W_regularizer=l1_l2(l1=regpara,l2=0.0),b_regularizer=l1_l2(l1=regpara,l2=0.0))(x)
    x = Dropout(dopara)(x)
    out = Dense(numofc, activation='softmax', name='output')(x)
    
    model = Model(input_img,out)
    model.summary()
    return model

if __name__ == '__main__':
    
    X = np.zeros((10,1,15,1))
    
    dmlp(X)
    dense_mlp(Xt=X)