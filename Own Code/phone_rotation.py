import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the path to the data directory
base_path = "Data_Arthur"


# Function to process each CSV file
def process_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Rename columns for convenience
    df.columns = ["Time", "Acceleration_x", "Acceleration_y", "Acceleration_z"]

    # Define the time intervals for aggregation (every half second)
    df['Time_interval'] = (df['Time'] // 0.5).astype(int)

    # Aggregate the data for every half second using the mean
    aggregated_df = df.groupby('Time_interval').mean().reset_index()

    # Calculate mean and standard deviation of the aggregated values
    mean_values = aggregated_df.mean()
    stdev_values = aggregated_df.std()

    return mean_values, stdev_values


# Initialize dictionaries to store results
all_mean_values = {}
all_stdev_values = {}

# Iterate over all folders in the base directory
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        file_path = os.path.join(folder_path, "Accelerometer.csv")
        if os.path.exists(file_path):
            mean_values, stdev_values = process_csv(file_path)

            # Store results in dictionaries
            all_mean_values[folder] = mean_values
            all_stdev_values[folder] = stdev_values

# Print the results
print("Mean and Standard Deviation of Aggregated Values for Each Folder:\n")
for folder in all_mean_values:
    print(f"Folder: {folder}")
    print("Mean values:")
    print(all_mean_values[folder])
    print("Standard Deviation values:")
    print(all_stdev_values[folder])
    print("\n" + "-" * 50 + "\n")

# Prepare data for plotting
folders = list(all_mean_values.keys())
mean_x = [all_mean_values[folder]['Acceleration_x'] for folder in folders]
mean_y = [all_mean_values[folder]['Acceleration_y'] for folder in folders]
mean_z = [all_mean_values[folder]['Acceleration_z'] for folder in folders]
stdev_x = [all_stdev_values[folder]['Acceleration_x'] for folder in folders]
stdev_y = [all_stdev_values[folder]['Acceleration_y'] for folder in folders]
stdev_z = [all_stdev_values[folder]['Acceleration_z'] for folder in folders]

# Create bar plots for mean values
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.bar(folders, mean_x, yerr=stdev_x, capsize=5, color='r', alpha=0.6)
plt.ylabel('Acceleration x (m/s^2)')
plt.title('Mean and Standard Deviation of Aggregated Acceleration x')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.bar(folders, mean_y, yerr=stdev_y, capsize=5, color='g', alpha=0.6)
plt.ylabel('Acceleration y (m/s^2)')
plt.title('Mean and Standard Deviation of Aggregated Acceleration y')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.bar(folders, mean_z, yerr=stdev_z, capsize=5, color='b', alpha=0.6)
plt.ylabel('Acceleration z (m/s^2)')
plt.xlabel('Folders')
plt.title('Mean and Standard Deviation of Aggregated Acceleration z')
plt.xticks(rotation=45)
plt.grid(True)


# Show plot
plt.show()