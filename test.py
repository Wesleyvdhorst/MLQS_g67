file_path = 'Data/Acceleration with g 2024-06-05 07-13-28-auto/Raw Data.csv'

import numpy as np
import matplotlib.pyplot as plt

# Load the data into a numpy array
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

# Extract columns
time = data[:, 0]
acc_x = data[:, 1]
acc_y = data[:, 2]
acc_z = data[:, 3]
abs_acc = data[:, 4]

# Define a function to compute the moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Window size for the moving average
window_size = 30

# Compute the moving averages
acc_x_ma = moving_average(acc_x, window_size)
acc_y_ma = moving_average(acc_y, window_size)
acc_z_ma = moving_average(acc_z, window_size)
abs_acc_ma = moving_average(abs_acc, window_size)

# Adjust the time array to match the length of the moving average arrays
time_ma = time[window_size - 1:]

# Plot the moving averages
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(time_ma, acc_x_ma, label='Moving Average Acceleration x (m/s^2)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration x (m/s^2)')
plt.title('Moving Average Acceleration x vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(time_ma, acc_y_ma, label='Moving Average Acceleration y (m/s^2)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration y (m/s^2)')
plt.title('Moving Average Acceleration y vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(time_ma, acc_z_ma, label='Moving Average Acceleration z (m/s^2)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration z (m/s^2)')
plt.title('Moving Average Acceleration z vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(time_ma, abs_acc_ma, label='Moving Average Absolute Acceleration (m/s^2)', color='k')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Acceleration (m/s^2)')
plt.title('Moving Average Absolute Acceleration vs Time')
plt.grid(True)
plt.legend()

# Adjust layout
plt.tight_layout()
plt.show()
