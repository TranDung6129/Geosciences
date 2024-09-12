import os
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy as sp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "0"
dataset_path = ""

def rotate_accelerometer_vectorized(acc, roll_error, pitch_error):
    """
    Rotate the accelerometer data to correct for roll and pitch errors.

    Parameters:
    acc (numpy.ndarray): The accelerometer data as a 2D numpy array with shape (N, 3),
                         where N is the number of samples and 3 represents the x, y, and z axes.
    roll_error (float): The roll error in radians.
    pitch_error (float): The pitch error in radians.

    Returns:
    numpy.ndarray: The rotated accelerometer data as a 2D numpy array with the same shape as the input.

    """
    # Create 3D rotation matrix for roll error correction
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll_error), -np.sin(roll_error)],
                   [0, np.sin(roll_error), np.cos(roll_error)]])

    # Create 3D rotation matrix for pitch error correction
    Ry = np.array([[np.cos(pitch_error), 0, np.sin(pitch_error)], 
                   [0, 1, 0],
                   [-np.sin(pitch_error), 0, np.cos(pitch_error)]])
    
    # Rotate the accelerometer data to correct for roll and pitch errors
    acc_corrected = np.matmul(np.matmul(Rx, Ry), acc.T).T
    return acc_corrected

def change_coordinate_frame(acc, roll, pitch, yaw):
    """
    Transforms the accelerometer data from the sensor frame to the body frame.

    Parameters:
    acc (numpy.ndarray): The accelerometer data in the sensor frame. Shape: (N, 3)
    roll (float): The roll angle in radians.
    pitch (float): The pitch angle in radians.
    yaw (float): The yaw angle in radians.

    Returns:
    numpy.ndarray: The transformed accelerometer data in the body frame. Shape: (N, 3)
    """
    # Create 3D rotation matrix for roll correction
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    # Create 3D rotation matrix for pitch correction
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], 
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    # Create 3D rotation matrix for yaw correction
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    # Rotate the accelerometer data to align with the body frame
    acc_transformed = np.matmul(np.matmul(np.matmul(Rx,Ry), Rz), acc.T).T
    return acc_transformed

def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    """
    Interpolates a 3D vector linearly based on input and output timestamps.

    Parameters:
    input (ndarray): The input 3D vector array.
    input_timestamp (ndarray): The timestamps corresponding to the input vector array.
    output_timestamp (ndarray): The timestamps for which the output vector needs to be interpolated.

    Returns:
    ndarray: The interpolated 3D vector array.

    Raises:
    AssertionError: If the shape of the input vector array and input timestamps array do not match.

    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0, kind='linear', fill_value='extrapolate')
    interpolate = func(output_timestamp)
    return interpolate

def BROAD_path():
    ''' i = [1,40]
    '''
    imu_path = []
    for i in range(1, 40):
        imu_path.append('trial_imu{}'.format(i))
    #gt_path = dataset_path + 'BROAD/trial_gt{}.csv'.format(i)
    #imu_path = dataset_path + 'BROAD/trial_imu{}.csv'.format(i)
    return imu_path

def BROAD_data(path):
    path = dataset_path+'BROAD/'+ path+'.csv'
    fs = 286
    imu_filename = path
    df = pd.read_csv(imu_filename, header=0).values
    acc = df[:, 0:3]
    gyro = df[:, 3:6]
    mag = df[:, 6:9]
    gt_filename = imu_filename.replace('imu', 'gt')
    df = pd.read_csv(gt_filename, header=0).values
    quat = df[:, 3:7]
    pose = df[:, 0:3]
    return acc, gyro, mag, quat,  fs