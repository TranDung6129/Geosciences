
import numpy as np
import pandas as pd

def identify_drilling_state(imu_data):
  """
  This function identifies the drilling state based on IMU data.

  Args:
      imu_data: A numpy array of shape (num_samples, 3) containing IMU data (accelerometer readings in x, y, z axes).

  Returns:
      A string indicating the drilling state ("drilling" or "not_drilling").
  """

  # Calculate magnitude of acceleration
  magnitude = np.linalg.norm(imu_data, axis=1)

  # Calculate standard deviation of magnitude
  std_mag = np.std(magnitude)

  # Define threshold for drilling state classification
  threshold = 0.5  # Adjust this value based on your data

  if std_mag > threshold:
    return "drilling"
  else:
    return "not_drilling"
  
# Read the text file into a DataFrame without header row and with specific column names
df = pd.read_csv("D:/Aitogy Projects/Drilling Identifier/data/lh3imu.txt", sep=' ', header=None, names=['timestamp', 'Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az'])

# Display the DataFrame
imu_data = df[['Ax', 'Ay', 'Az']].values

# Identify drilling state
drilling_state = identify_drilling_state(imu_data)

print(drilling_state)
