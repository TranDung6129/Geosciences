import tensorflow as tf

import tensorflow_addons as tfa
import keras.backend as K

from keras.layers import *
from tensorflow import keras

import matplotlib.pyplot as plt
from fileinput import filename
import pandas as pd
import random as rn
import numpy as np
import math
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['PYTHONHASHSEED'] = '0'

def AttLayer(q):
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    roll = tf.math.atan2(2*(w*x + y*z),
                       (1-2*(x*x + y*y)))
    pitch = tf.math.asin(2*(w*y - x*z))
    zero_float64 = tf.constant(0.0, dtype=tf.float64)
    qx = (tf.math.sin(roll/2) * tf.math.cos(pitch/2) * tf.math.cos(zero_float64) -
          tf.math.cos(roll/2) * tf.math.sin(pitch/2) * tf.math.sin(zero_float64))
    qx = tf.reshape(qx, (tf.shape(roll)[0], 1))
    qy = (tf.math.cos(roll/2) * tf.math.sin(pitch/2) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.cos(pitch/2) * tf.math.sin(zero_float64))
    qy = tf.reshape(qy, (tf.shape(roll)[0], 1))
    qz = (tf.math.cos(roll/2) * tf.math.cos(pitch/2) * tf.math.sin(zero_float64) -
          tf.math.sin(roll/2) * tf.math.sin(pitch/2) * tf.math.cos(zero_float64))
    qz = tf.reshape(qz, (tf.shape(roll)[0], 1))
    qw = (tf.math.cos(roll/2) * tf.math.cos(pitch/2) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.sin(pitch/2) * tf.math.sin(zero_float64))
    qw = tf.reshape(qw, (tf.shape(roll)[0], 1))
    quat = tf.concat([qw, qx, qy, qz], axis=-1)
    return quat

def Model_B(window_size=200):
    Gn = 0.25
    acc = Input((window_size, 3), name='Acc')
    Acc = GaussianNoise(Gn, name='GaussianNoiseAcc')(acc)
    
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    A1LSTM = Bidirectional(LSTM(50, return_sequences=True, name='A1LSTM'))(Acc1)
    A1LSTM = Dropout(0.2, name='DropoutA1LSTM')(A1LSTM)

    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    A2LSTM = Bidirectional(LSTM(50, return_sequences=True, name='A2LSTM'))(Acc2)
    A2LSTM = Dropout(0.2, name='DropoutA2LSTM')(A2LSTM)

    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)
    A3LSTM = Bidirectional(LSTM(50, return_sequences=True, name='A3LSTM'))(Acc3)
    A3LSTM = Dropout(0.2, name='DropoutA3LSTM')(A3LSTM)

    Aconc = concatenate([A1LSTM, A2LSTM, A3LSTM], name='Aconc')
    Aconc = Flatten(name='FlattenAconc')(Aconc)

    gyro = Input((window_size, 3), name='Gyro')
    Gyro = GaussianNoise(Gn, name='GaussianNoiseGyro')(gyro)
    
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    G1LSTM = Bidirectional(LSTM(50, return_sequences=True, name='G1LSTM'))(Gyro1)
    G1LSTM = Dropout(0.2, name='DropoutG1LSTM')(G1LSTM)

    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    G2LSTM = Bidirectional(LSTM(50, return_sequences=True, name='G2LSTM'))(Gyro2)
    G2LSTM = Dropout(0.2, name='DropoutG2LSTM')(G2LSTM)

    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)
    G3LSTM = Bidirectional(LSTM(50, return_sequences=True, name='G3LSTM'))(Gyro3)
    G3LSTM = Dropout(0.2, name='DropoutG3LSTM')(G3LSTM)

    Gconc = concatenate([G1LSTM, G2LSTM, G3LSTM], name='Gconc')
    Gconc = Flatten(name='FlattenGconc')(Gconc)

    conc = concatenate([Aconc, Gconc], name='Conc')
    conc = Dense(256, activation="relu", name='DenseConc')(conc)

    fs = Input((1,), name='Fs')
    fsDense = Dense(conc.shape[1], activation="relu", name='FsDense')(fs)

    conc2 = concatenate([conc, fsDense], name='Conc2')
    conc2 = Dense(256, activation="relu", name='DenseConc2')(conc2)
    conc2 = GaussianNoise(Gn, name='GaussianNoiseConc2')(conc2)
    quat = Dense(4, activation="linear", name='Quat')(conc2)
    quat = Lambda(lambda x: AttLayer(x), name='Attitude')(quat)
    quat = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='NormalizeQuat')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=[quat])
    model.summary()
    return model

def roll_from_quaternion_layer(q):
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    roll = tf.math.atan2(2*(w*x + y*z),
                       (1-2*(x*x + y*y)))
    pitch = tf.math.asin(2*(w*y - x*z))
    zero_float64 = tf.constant(0.0, dtype=tf.float64)
    qx = (tf.math.sin(roll/2) * tf.math.cos(zero_float64) * tf.math.cos(zero_float64) -
          tf.math.cos(roll/2) * tf.math.sin(zero_float64) * tf.math.sin(zero_float64))
    qx = tf.reshape(qx, (tf.shape(roll)[0], 1))
    qy = (tf.math.cos(roll/2) * tf.math.sin(zero_float64) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.cos(zero_float64) * tf.math.sin(zero_float64))
    qy = tf.reshape(qy, (tf.shape(roll)[0], 1))
    qz = (tf.math.cos(roll/2) * tf.math.cos(zero_float64) * tf.math.sin(zero_float64) -
          tf.math.sin(roll/2) * tf.math.sin(zero_float64) * tf.math.cos(zero_float64))
    qz = tf.reshape(qz, (tf.shape(roll)[0], 1))
    qw = (tf.math.cos(roll/2) * tf.math.cos(zero_float64) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.sin(zero_float64) * tf.math.sin(zero_float64))
    qw = tf.reshape(qw, (tf.shape(roll)[0], 1))
    quat = tf.concat([qw, qx, qy, qz], axis=-1)
    return quat

def pitch_from_quaternion_layer(q):
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    zero_float64 = tf.constant(0.0, dtype=tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    pitch = tf.math.asin(-2*(w*y - x*z))
    quat_pitch = tf.concat([tf.cos(pitch/2), tf.zeros_like(pitch), tf.sin(pitch/2), tf.zeros_like(pitch)], axis=-1)
    return pitch
def yaw_from_quaternion_layer(q):
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    yaw = tf.math.atan2(2*(w*x + y*z), (w*w - x*x - y*y + z*z))
    quat_yaw = tf.concat([tf.cos(yaw/2), tf.zeros_like(yaw), tf.zeros_like(yaw), tf.sin(yaw/2)], axis=-1)
    return quat_yaw

def roll_est(window_size):
    Gn = 0.25
    acc = Input((window_size, 3), name='acc')
    Acc = GaussianNoise(Gn, name='GaussianNoiseAcc')(acc)
    gyro = Input((window_size, 3), name='gyro')
    Gyro = GaussianNoise(Gn, name='GaussianNoiseGyro')(gyro)
    fs = Input((1,), name='fs')
    Fs = GaussianNoise(0.1, name='GaussianNoiseFs')(fs)
    
    ALSTM = Bidirectional(LSTM(128,
                                   return_sequences=True,
                                   # return_state=True,
                                   #go_backwards=True,
                                   name='BiLSTM1'))(AGconcat)
    
    
    GLSTM = Bidirectional(LSTM(128,
                                   return_sequences=True,
                                   # return_state=True,
                                   #go_backwards=True,
                                   name='BiLSTM2'))(AGconcat)
    
    AGconcat = concatenate([Acc, Gyro])
    AGLSTM = Bidirectional(LSTM(128,
                                   return_sequences=True,
                                   # return_state=True,
                                   #go_backwards=True,
                                   name='BiLSTM3'))(AGconcat)
    
    Fdense = Dense(units=256,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    x = Dense(units=256,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    quat = Dense(4, activation='linear', name='quat')(x)
    quat = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='NormalizeQuat')(quat)
    quat = Lambda(roll_from_quaternion_layer, name='Attitude')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=quat)
    model.summary()
    return model
