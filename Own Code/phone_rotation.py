import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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
    #df = apply_kalman_filter(df)

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
                filename = f"{data_type}.csv"
                save_path = os.path.join(instance_folder, filename)
                df.to_csv(save_path, index=False)

# Group the results by mode of transportation
mean_values_by_mode = {}
stdev_values_by_mode = {}

for folder in all_mean_values:
    mode = get_mode_of_transportation(folder)
    if mode not in mean_values_by_mode:
        mean_values_by_mode[mode] = {}
        stdev_values_by_mode[mode] = {}

    mean_values_by_mode[mode][folder] = all_mean_values[folder]
    stdev_values_by_mode[mode][folder] = all_stdev_values[folder]

# Function to plot grouped by mode
def plot_grouped_by_mode(show_plots=True, filtered=False):
    if not show_plots:
        return

    for mode in mean_values_by_mode:
        for data_type in ["Accelerometer", "Gyroscope", "Linear Accelerometer"]:
            folders = list(mean_values_by_mode[mode].keys())
            mean_x = [mean_values_by_mode[mode][folder][data_type]['Acceleration_x'] for folder in folders]
            mean_y = [mean_values_by_mode[mode][folder][data_type]['Acceleration_y'] for folder in folders]
            mean_z = [mean_values_by_mode[mode][folder][data_type]['Acceleration_z'] for folder in folders]
            stdev_x = [stdev_values_by_mode[mode][folder][data_type]['Acceleration_x'] for folder in folders]
            stdev_y = [stdev_values_by_mode[mode][folder][data_type]['Acceleration_y'] for folder in folders]
            stdev_z = [stdev_values_by_mode[mode][folder][data_type]['Acceleration_z'] for folder in folders]

            # Create bar plots for mean values
            plt.figure(figsize=(14, 8))

            plt.subplot(3, 1, 1)
            plt.bar(folders, mean_x, yerr=stdev_x, capsize=5, color='r', alpha=0.6)
            plt.ylabel('Acceleration x (m/s^2)')
            if filtered:
                plt.title(f'Mean and Standard Deviation of Filtered Acceleration x for {mode} - {data_type}')
            else:
                plt.title(f'Mean and Standard Deviation of Original Acceleration x for {mode} - {data_type}')
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.bar(folders, mean_y, yerr=stdev_y, capsize=5, color='g', alpha=0.6)
            plt.ylabel('Acceleration y (m/s^2)')
            if filtered:
                plt.title(f'Mean and Standard Deviation of Filtered Acceleration y for {mode} - {data_type}')
            else:
                plt.title(f'Mean and Standard Deviation of Original Acceleration y for {mode} - {data_type}')
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.bar(folders, mean_z, yerr=stdev_z, capsize=5, color='b', alpha=0.6)
            plt.ylabel('Acceleration z (m/s^2)')
            plt.xlabel('Folders')
            if filtered:
                plt.title(f'Mean and Standard Deviation of Filtered Acceleration z for {mode} - {data_type}')
            else:
                plt.title(f'Mean and Standard Deviation of Original Acceleration z for {mode} - {data_type}')
            plt.xticks(rotation=45)
            plt.grid(True)

            # Show plot
            plt.tight_layout()
            plt.show()

# Function to plot all data in one figure
def plot_all_in_one(show_plots=True, filtered=False):
    if not show_plots:
        return

    for data_type in ["Accelerometer", "Gyroscope", "Linear Accelerometer"]:
        folders = list(all_mean_values.keys())
        mean_x = [all_mean_values[folder][data_type]['Acceleration_x'] for folder in folders]
        mean_y = [all_mean_values[folder][data_type]['Acceleration_y'] for folder in folders]
        mean_z = [all_mean_values[folder][data_type]['Acceleration_z'] for folder in folders]
        stdev_x = [all_stdev_values[folder][data_type]['Acceleration_x'] for folder in folders]
        stdev_y = [all_stdev_values[folder][data_type]['Acceleration_y'] for folder in folders]
        stdev_z = [all_stdev_values[folder][data_type]['Acceleration_z'] for folder in folders]

        # Create bar plots for mean values
        plt.figure(figsize=(14, 8))

        plt.subplot(3, 1, 1)
        plt.bar(folders, mean_x, yerr=stdev_x, capsize=5, color='r', alpha=0.6)
        plt.ylabel('Acceleration x (m/s^2)')
        if filtered:
            plt.title(f'Mean and Standard Deviation of Filtered Acceleration x for All Modes - {data_type}')
        else:
            plt.title(f'Mean and Standard Deviation of Original Acceleration x for All Modes - {data_type}')
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.bar(folders, mean_y, yerr=stdev_y, capsize=5, color='g', alpha=0.6)
        plt.ylabel('Acceleration y (m/s^2)')
        if filtered:
            plt.title(f'Mean and Standard Deviation of Filtered Acceleration y for All Modes - {data_type}')
        else:
            plt.title(f'Mean and Standard Deviation of Original Acceleration y for All Modes - {data_type}')
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.bar(folders, mean_z, yerr=stdev_z, capsize=5, color='b', alpha=0.6)
        plt.ylabel('Acceleration z (m/s^2)')
        plt.xlabel('Folders')
        if filtered:
            plt.title(f'Mean and Standard Deviation of Filtered Acceleration z for All Modes - {data_type}')
        else:
            plt.title(f'Mean and Standard Deviation of Original Acceleration z for All Modes - {data_type}')
        plt.xticks(rotation=45)
        plt.grid(True)

        # Show plot
        plt.tight_layout()
        plt.show()


# Print the results
def print_results():
    print("Mean and Standard Deviation of Aggregated Values for Each Folder:\n")
    for folder in all_mean_values:
        print(f"Folder: {folder}")
        print("Accelerometer - Mean values:")
        print(all_mean_values[folder]["Accelerometer"])
        print("Gyroscope - Mean values:")
        print(all_mean_values[folder]["Gyroscope"])
        print("Linear Accelerometer - Mean values:")
        print(all_mean_values[folder]["Linear Accelerometer"])

        print("Accelerometer - Standard Deviation values:")
        print(all_stdev_values[folder]["Accelerometer"])
        print("Gyroscope - Standard Deviation values:")
        print(all_stdev_values[folder]["Gyroscope"])
        print("Linear Accelerometer - Standard Deviation values:")
        print(all_stdev_values[folder]["Linear Accelerometer"])

        print("\n" + "-" * 50 + "\n")

# Example calls to the functions
print_results()
plot_grouped_by_mode(show_plots=True, filtered=True)
plot_all_in_one(show_plots=False, filtered=True)
