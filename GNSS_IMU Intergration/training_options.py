import argparse
import math
import os
import random as rn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from keras.callbacks import LambdaCallback
from tqdm.keras import TqdmCallback

from model import *
from load_data import *
from learning_process import *
from utils import *

import inspect
from inspect import getmembers, isfunction

# Get current timestamp
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

# Set random seeds for reproducibility
np.random.seed(0)
rn.seed(1254)
tf.random.set_seed(89)

# Initialize variables
imu_data = []
epochs = 300
batch_size = 520
lr = 0.0001 * (batch_size / 32)
window_size = 100
stride = 4
if window_size == 1:
    stride = 1
version = 1
pred_model = Model_B

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1) 
    
def batchsize(number):
    batch_size = []
    for i in range(1,number):
        if number%i==0:
            batches.append(i)
    return (batches)

def learningRate(model):
    [acc, gyro, _, fs], [quat] = data_train(window_size, stride)
    acc, x_acc_val, gyro, x_gyro_val, fs, x_fs_val, quat, Att_quat_val = train_test_split(
        acc, gyro, fs, quat, test_size=0.2, random_state=42, shuffle=True)
    model = model(window_size)
    model.compile(optimizer=tfa.optimizers.RectifiedAdam(
        learning_rate=lr,), loss=QQuat_mult, metrics=[Quat_error_angle])
    # find best learning rate
    lr_finder = LRFinder(model)
    lr_finder.find([acc, gyro, fs], quat,
                   start_lr=1e-5,
                   end_lr=10, batch_size=batch_size, epochs=1)
    lr_finder.plot_loss()
    print("Learning rate finder complete")

def data_broad(window_size, stride):
    # broad data
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))

    #
    broad_set = [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16,
                 17, 18, 20, 21, 22, 23, 26, 28, 29, 30, 38, 39]
    #broad_set = np.arange(1, 37)
    for i in broad_set:
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = BROAD_data(
            BROAD_path()[i-1])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]

def data_train(window_size, stride):
    [gyro_broad, acc_broad, mag_broad, fs_broad], [
        quat_broad] = data_broad(window_size, stride)
    print("broad done", gyro_broad.shape, acc_broad.shape,
          mag_broad.shape, fs_broad.shape, quat_broad.shape)
    return [acc_broad, gyro_broad, mag_broad, fs_broad, ], [quat_broad]
def learningRate(model):
    [gyro, acc, mag, fs], [quat] = data_train(window_size, stride)
    acc_train, acc_val, quat_train, quat_val = train_test_split(acc, quat, test_size=0.2, random_state=42, shuffle=True)
    inputs = np.concatenate((gyro, acc_train, mag, fs), axis=1)
    outputs = quat_train
    model = model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=PILoss)
    model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs, validation_data=(inputs_val, outputs_val), callbacks=[early_stopping, lr_scheduler])

def train(pred_model):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # training callbacks
    # Checkpoint
    [x_acc, x_gyro, x_mag, x_fs], [quat] = data_train(window_size, stride)
    quat = Att(quat)
    # find the factors of len(quat)
    print("quat", quat.shape)
    x_acc, x_gyro, x_mag, x_fs, quat = x_acc, x_gyro, x_mag, x_fs, quat
    x_acc, x_gyro, x_mag, x_fs, quat = shuffle(x_acc, x_gyro, x_mag, x_fs, quat)

    lr = 0.0001 *(batch_size/32)
    #x_acc, x_gyro = x_acc.reshape(x_acc.shape[0], 1,x_acc.shape[1]), x_gyro.reshape(x_gyro.shape[0], 1,x_gyro.shape[1])
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    quat = quaternion_roll(quat)
    model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5'
        , monitor='val_loss', verbose=1, save_best_only=True)
    
    # tensorboard
    tensorboard = TensorBoard(log_dir="./logs/%s_%s" % (timestr,
                                                        pred_model.__name__),
                              histogram_freq=0, write_graph=True, write_images=True)
    # EarlyStopping
    earlystopping = EarlyStopping(
        monitor='val_loss', patience=25, verbose=1)
    # ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=20, min_lr=0.0000001)
    # keras CosineDecay
    lr_scheduleCosineDec = tf.keras.experimental.CosineDecay(
        initial_learning_rate=lr, decay_steps=epochs)
    CosineDecay = LearningRateScheduler(lr_scheduleCosineDec)
    # keras CosineDecayRestarts
    lr_scheduleCosineDecRest = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate=lr, first_decay_steps=5)
    CosineDecayRestarts = LearningRateScheduler(lr_scheduleCosineDecRest)
    # CSVLogger
    csv_logger = CSVLogger('training_%s.log'% pred_model.__name__) 
    print("Training model: ", pred_model.__name__)
    callbacklist = [model_checkpoint, tensorboard,
                    earlystopping,reduce_lr]
    # shuffle data for training TqdmCallback(verbose=2)
    print("Learning rate: ", lr)
    model = pred_model(window_size)
    # truncated backpropagation through time
    model.compile(optimizer=Adam(learning_rate=lr)
                  , loss= QQuat_mult, metrics=[Quat_error_angle])
    
    history = model.fit(
        [x_acc, x_gyro,x_fs],
        quat,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[callbacklist],
        validation_split=0.2,
        verbose=1,
        shuffle=True,)
    model.save('%s_B%s_E%s_V%s.hdf5' %
               (pred_model.__name__, batch_size, epochs, version))

    # plot training history
    plt.figure(figsize=(15, 15))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
        
    plt.grid(color='white', linestyle='--', linewidth=0.8)
    
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.savefig(f'loss_%s_B%s_E%s_V%s.png' %
                (pred_model.__name__, batch_size, epochs, version))
    # plt.show()
    # plot learning rate
    plt.figure(figsize=(15, 15))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)    
    plt.grid(color='white', linestyle='--', linewidth=0.8)
    plt.plot(history.history['lr'])
    plt.title('Learning rate', fontsize=20)
    plt.ylabel('Learning rate', fontsize=15)
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(f'lr_%s_B%s_E%s_V%s.png' %
                (pred_model.__name__, batch_size, epochs, version))
    
def main():
    train(pred_model)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
