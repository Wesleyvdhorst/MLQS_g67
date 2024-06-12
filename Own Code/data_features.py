import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.fft import fft

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


def pca_features(df, n_components=3):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df[['X', 'Y', 'Z']])
    for i in range(1, n_components + 1):
        df[f'pca{i}'] = pca_result[:, i - 1]
    return df


def statistical_features(df, window_size=10):
    # Moving window feature calculation
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


def frequency_features(df):
    for ax in ['X', 'Y', 'Z']:
        # Apply FFT
        fft_values = fft(df[ax].values)
        df[f'{ax}_fft'] = np.abs(fft_values)

        # Extract specific frequency domain features
        df[f'{ax}_dominant_freq'] = np.argmax(np.abs(fft_values))
        df[f'{ax}_spectral_energy'] = np.sum(np.abs(fft_values) ** 2)
        df[f'{ax}_spectral_entropy'] = -np.sum(np.abs(fft_values) * np.log(np.abs(fft_values)))

    return df


def feature_engineering(file_path, sensor):
    filtered_df = pd.read_csv(file_path)

    # Create PCA Features
    pca_df = pca_features(filtered_df.copy(), n_components=3)
    all_dataframes[folder][sensor]['pca'] = pca_df

    # Create Statistical Features
    pca_time_df = statistical_features(pca_df.copy())
    all_dataframes[folder][sensor]['pca_time'] = pca_time_df

    # Create Frequency Features (Fourier Transformation)
    pca_time_freq_df = frequency_features(pca_time_df.copy())
    all_dataframes[folder][sensor]['pca_time_freq'] = pca_time_freq_df

    return pca_time_freq_df


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
            for sensor in sensor_types:
                all_dataframes[folder][sensor] = {}

        # Apply Feature Engineering
        feature_engineering(accelerometer_path, sensor_types[0])
        feature_engineering(gyroscope_path, sensor_types[1])
        feature_engineering(linear_path, sensor_types[2])

        # Create a folder for the instance
        instance_folder = os.path.join(feature_data_path, f"{mode}_{mode_counts[mode]}")
        if not os.path.exists(instance_folder):
            os.makedirs(instance_folder)

        # Save each DataFrame to CSV within the instance folder
        for sensor in sensor_types:
            for step, feature_df in all_dataframes[folder][sensor].items():
                filename = f"{sensor}_{step}.csv"
                save_path = os.path.join(instance_folder, filename)
                feature_df.to_csv(save_path, index=False)
