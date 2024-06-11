import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sensor_types = ['Accelerometer', 'Gyroscope', 'Linear Accelerometer']
filtered_data_path = "Data_Filtered"
feature_data_path = "Data_Features"

# Create the filtered data directory if it doesn't exist
if not os.path.exists(feature_data_path):
    os.makedirs(feature_data_path)


def get_mode_of_transportation(folder_name):
    return folder_name.split('_')[0]


# Dictionary to keep track of the number of instances of each mode of transportation
mode_counts = {}
all_dataframes = {}


def feature_engineering(file_path):
    df = pd.read_csv(file_path)

    # Calculate rolling mean over a 10-second window
    window_size = 10  # 10 seconds window

    # Rolling mean calculation
    for ax in ['X', 'Y', 'Z']:
        df[f'temp_{ax}_mean'] = df[ax].rolling(window=window_size).mean()
        df[f'temp_{ax}_std'] = df[ax].rolling(window=window_size).std()
        df[f'temp_{ax}_median'] = df[ax].rolling(window=window_size).median()
        df[f'temp_{ax}_sum'] = df[ax].rolling(window=window_size).sum()
        df[f'temp_{ax}_skew'] = df[ax].rolling(window=window_size).skew()
        df[f'temp_{ax}_kurt'] = df[ax].rolling(window=window_size).kurt()

        # Calculate lag features
        df[f'{ax}_lag1'] = df[ax].shift(1)
        df[f'{ax}_lag2'] = df[ax].shift(2)

    return df


for folder in os.listdir(filtered_data_path):
    folder_path = os.path.join(filtered_data_path, folder)
    if os.path.isdir(folder_path):
        # Read filtered sensor type dataframes
        accelerometer_path = os.path.join(folder_path, f"{sensor_types[0]}.csv")
        gyroscope_path = os.path.join(folder_path, f"{sensor_types[1]}.csv")
        linear_path = os.path.join(folder_path, f"{sensor_types[2]}.csv")

        # Extract mode of transportation
        mode = get_mode_of_transportation(folder)

        # Increment instance count for the mode of transportation
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Initialize the dictionary for the current folder if not already initialized
        if folder not in all_dataframes:
            all_dataframes[folder] = {}

        # Apply Feature Engineering
        feature_acc = feature_engineering(accelerometer_path)
        all_dataframes[folder][sensor_types[0]] = feature_acc

        feature_gyro = feature_engineering(gyroscope_path)
        all_dataframes[folder][sensor_types[1]] = feature_gyro

        feature_linear = feature_engineering(linear_path)
        all_dataframes[folder][sensor_types[2]] = feature_linear

        # Create a folder for the instance
        instance_folder = os.path.join(feature_data_path, f"{mode}_{mode_counts[mode]}")
        if not os.path.exists(instance_folder):
            os.makedirs(instance_folder)

        # Save each DataFrame to CSV within the instance folder
        for data_type, df in all_dataframes[folder].items():
            filename = f"{data_type}.csv"
            save_path = os.path.join(instance_folder, filename)
            df.to_csv(save_path, index=False)
