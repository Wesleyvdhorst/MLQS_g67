import pandas as pd
import numpy as np
import os
from filterpy.kalman import KalmanFilter

# Define the path to the data directory
base_paths = ["Data_Arthur", "Data_Nando"]
filtered_data_path = "Data_Filtered"

# Create the filtered data directory if it doesn't exist
if not os.path.exists(filtered_data_path):
    os.makedirs(filtered_data_path)

# Function to apply a Kalman filter to the data
def apply_kalman_filter(df):
    # Initialize Kalman filters for each axis
    kf_x = KalmanFilter(dim_x=2, dim_z=1)
    kf_y = KalmanFilter(dim_x=2, dim_z=1)
    kf_z = KalmanFilter(dim_x=2, dim_z=1)

    # Define initial parameters for the filters
    for kf in [kf_x, kf_y, kf_z]:
        kf.x = np.array([0., 0.])  # Initial state
        kf.F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
        kf.H = np.array([[1., 0.]])  # Measurement function
        kf.P *= 1000.  # Initial uncertainty
        kf.R = 5  # Measurement noise
        kf.Q = 0.1  # Process noise

    # Apply the Kalman filter to each acceleration component
    filtered_x = []
    filtered_y = []
    filtered_z = []

    for i in range(len(df)):
        kf_x.predict()
        kf_x.update(df['Acceleration_x'][i])
        filtered_x.append(kf_x.x[0])

        kf_y.predict()
        kf_y.update(df['Acceleration_y'][i])
        filtered_y.append(kf_y.x[0])

        kf_z.predict()
        kf_z.update(df['Acceleration_z'][i])
        filtered_z.append(kf_z.x[0])

    # Update the DataFrame with the filtered values
    df['Acceleration_x'] = filtered_x
    df['Acceleration_y'] = filtered_y
    df['Acceleration_z'] = filtered_z

    return df


# Function to extract the mode of transportation from the folder name
def get_mode_of_transportation(folder_name):
    return folder_name.split('_')[0]


# Function to process each CSV file
def process_csv(file_path, y_aligned_with_gravity):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Rename columns for convenience
    df.columns = ["Time", "Acceleration_x", "Acceleration_y", "Acceleration_z"]

    # Apply the Kalman filter to the data
    df = apply_kalman_filter(df)

    # If y is aligned with gravity, switch y and z in this file as well
    if y_aligned_with_gravity:
        df.rename(columns={"Acceleration_y": "Temp", "Acceleration_z": "Acceleration_y"}, inplace=True)
        df.rename(columns={"Temp": "Acceleration_z"}, inplace=True)

    # Define the time intervals for aggregation (every half second)
    df['Time_interval'] = (df['Time'] // 0.5).astype(int)

    # Aggregate the data for every half second using the mean
    aggregated_df = df.groupby('Time_interval').mean().reset_index()

    # Calculate mean and standard deviation of the aggregated values
    mean_values = aggregated_df.mean()
    stdev_values = aggregated_df.std()

    return aggregated_df, mean_values, stdev_values

# Initialize dictionaries to store results
all_mean_values = {}
all_stdev_values = {}
all_dataframes = {}

# Dictionary to keep track of the number of instances of each mode of transportation
mode_counts = {}

# Iterate over all folders in the base directory
for base_path in base_paths:
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            accelerometer_path = os.path.join(folder_path, "Accelerometer.csv")
            gyroscope_path = os.path.join(folder_path, "Gyroscope.csv")

            # Check for both possible names of the third file
            linear_accelerometer_path = os.path.join(folder_path, "Linear Accelerometer.csv")
            linear_acceleration_path = os.path.join(folder_path, "Linear Acceleration.csv")

            # Determine the correct file path for the third file
            if os.path.exists(linear_accelerometer_path):
                linear_path = linear_accelerometer_path
            elif os.path.exists(linear_acceleration_path):
                linear_path = linear_acceleration_path
            else:
                linear_path = None

            # Process each CSV file
            if os.path.exists(accelerometer_path):
                # Process the Accelerometer.csv file first to determine axis alignment
                accelerometer_df = pd.read_csv(accelerometer_path)
                accelerometer_df.columns = ["Time", "Acceleration_x", "Acceleration_y", "Acceleration_z"]
                median_acc_y = accelerometer_df["Acceleration_y"].median()
                median_acc_z = accelerometer_df["Acceleration_z"].median()
                y_aligned_with_gravity = abs(median_acc_y - 9.8) < abs(median_acc_z - 9.8)

                # Process the accelerometer data
                aggregated_acc_df, mean_acc, stdev_acc = process_csv(accelerometer_path, y_aligned_with_gravity)
                all_mean_values[folder] = {"Accelerometer": mean_acc}
                all_stdev_values[folder] = {"Accelerometer": stdev_acc}
                all_dataframes[folder] = {"Accelerometer": aggregated_acc_df}

            if os.path.exists(gyroscope_path):
                # Process the Gyroscope.csv file
                aggregated_gyro_df, mean_gyro, stdev_gyro = process_csv(gyroscope_path, y_aligned_with_gravity)
                all_mean_values[folder]["Gyroscope"] = mean_gyro
                all_stdev_values[folder]["Gyroscope"] = stdev_gyro
                all_dataframes[folder]["Gyroscope"] = aggregated_gyro_df

            if linear_path is not None:
                # Process the Linear Accelerometer.csv file
                aggregated_lin_acc_df, mean_lin_acc, stdev_lin_acc = process_csv(linear_path, y_aligned_with_gravity)
                all_mean_values[folder]["Linear Accelerometer"] = mean_lin_acc
                all_stdev_values[folder]["Linear Accelerometer"] = stdev_lin_acc
                all_dataframes[folder]["Linear Accelerometer"] = aggregated_lin_acc_df

            # Extract mode of transportation
            mode = folder.split('_')[0]

            # Increment instance count for the mode of transportation
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

            # Create a folder for the instance
            instance_folder = os.path.join(filtered_data_path, f"{mode}_{mode_counts[mode]}")
            if not os.path.exists(instance_folder):
                os.makedirs(instance_folder)

            # Save each DataFrame to CSV within the instance folder
            for data_type, df in all_dataframes[folder].items():
                # Drop the "Time Interval" column if present
                if "Time_interval" in df.columns:
                    df.drop(columns=["Time_interval"], inplace=True)
                # Change column names based on the data type
                if data_type == "Accelerometer":
                    df.columns = ["Time (s)", "X", "Y", "Z"]
                elif data_type == "Gyroscope":
                    df.columns = ["Time (s)", "X", "Y", "Z"]
                elif data_type == "Linear Accelerometer":
                    df.columns = ["Time (s)", "X", "Y", "Z"]

                # Save the DataFrame to CSV within the instance folder
                filename = f"{data_type}.csv"
                save_path = os.path.join(instance_folder, filename)
                df.to_csv(save_path, index=False)
