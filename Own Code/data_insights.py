import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to extract the mode of transportation from the folder name
def get_mode_of_transportation(folder_name):
    return folder_name.split('_')[0]

# Load the aggregated data from the filtered folder
def load_filtered_data(filtered_data_path):
    filtered_data = {}

    for folder in os.listdir(filtered_data_path):
        folder_path = os.path.join(filtered_data_path, folder)
        if os.path.isdir(folder_path):
            filtered_data[folder] = {}
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    data_type = file.split('.')[0]
                    df = pd.read_csv(file_path)
                    filtered_data[folder][data_type] = df

    return filtered_data

# Load the filtered data
filtered_data = load_filtered_data("Data_Filtered")

# Group the data by mode of transportation
mean_values_by_mode = {}
stdev_values_by_mode = {}

for folder, data in filtered_data.items():
    mode = get_mode_of_transportation(folder)
    if mode not in mean_values_by_mode:
        mean_values_by_mode[mode] = {}
        stdev_values_by_mode[mode] = {}

    for data_type, df in data.items():
        mean_values_by_mode[mode][data_type] = df.mean()
        stdev_values_by_mode[mode][data_type] = df.std()

# Function to plot grouped by mode
def plot_grouped_by_mode(show_plots=True, filtered=True):
    if not show_plots:
        return

    for mode in mean_values_by_mode:
        for data_type in mean_values_by_mode[mode]:
            mean_values = mean_values_by_mode[mode][data_type].drop("Time (s)", errors='ignore')
            stdev_values = stdev_values_by_mode[mode][data_type].drop("Time (s)", errors='ignore')

            plt.figure(figsize=(14, 6))

            plt.bar(mean_values.index, mean_values.values, yerr=stdev_values.values, capsize=5, color='b', alpha=0.6)
            plt.ylabel('Acceleration (m/s^2)')
            plt.xlabel('Data Type')
            plt.title(f'Mean and Standard Deviation of {"Filtered" if filtered else "Original"} {data_type} for {mode}')
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.tight_layout()
            plt.show()


def plot_all_modes_by_measurement(show_plots=True, filtered=True):
    if not show_plots:
        return

    measurements = ["Accelerometer", "Gyroscope", "Linear Accelerometer"]

    for measurement in measurements:
        fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        fig.suptitle(
            f'Mean and Standard Deviation of {"Filtered" if filtered else "Original"} {measurement} for All Modes')

        axes_labels = ["X", "Y", "Z"]
        bar_width = 0.2
        mode_names = list(mean_values_by_mode.keys())

        for i, axis in enumerate(axes_labels):
            index = np.arange(len(mode_names))

            for j, mode in enumerate(mode_names):
                mean_values = mean_values_by_mode[mode].get(measurement)
                stdev_values = stdev_values_by_mode[mode].get(measurement)

                if mean_values is not None and stdev_values is not None:
                    mean_values = mean_values.drop("Time (s)", errors='ignore')
                    stdev_values = stdev_values.drop("Time (s)", errors='ignore')

                    if axis in mean_values.index:
                        axs[i].bar(index + j * bar_width, mean_values[axis], yerr=stdev_values[axis], capsize=5,
                                   label=mode if i == 0 else "", width=bar_width, alpha=0.6)

            axs[i].set_xlabel('Mode')
            axs[i].set_ylabel(f'{axis} Acceleration (m/s^2)')
            axs[i].set_xticks(index + bar_width * (len(mode_names) - 1) / 2)
            axs[i].set_xticklabels(mode_names, rotation=45)
            axs[i].grid(True)
            axs[i].legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# Function to provide data insights
def provide_data_insights():
    for mode in mean_values_by_mode:
        print(f"Mode of Transportation: {mode}")
        print("=" * 50)
        for data_type in mean_values_by_mode[mode]:
            mean_values = mean_values_by_mode[mode][data_type]
            stdev_values = stdev_values_by_mode[mode][data_type]

            print(f"\nData Type: {data_type}")
            print("Mean Values:")
            print(mean_values)
            print("\nStandard Deviation Values:")
            print(stdev_values)
            print("-" * 50)


def plot_time_series(folder_path, label, interval_start=0, interval_end=30):
    # Load the CSV files for Accelerometer, Gyroscope, and Linear Accelerometer
    accelerometer_df = pd.read_csv(os.path.join(folder_path, "Accelerometer.csv"))
    gyroscope_df = pd.read_csv(os.path.join(folder_path, "Gyroscope.csv"))
    linear_accelerometer_df = pd.read_csv(os.path.join(folder_path, "Linear Accelerometer.csv"))

    # Extract data from the dataframes
    time = accelerometer_df.iloc[:, 0]

    # Find the indices corresponding to the specified time interval
    start_idx = (time - interval_start).abs().idxmin()
    end_idx = (time - interval_end).abs().idxmin()

    # Extract the data for the specified interval
    time_interval = time[start_idx:end_idx]

    # Extract accelerometer data
    x_accelerometer = accelerometer_df.iloc[start_idx:end_idx, 1]
    y_accelerometer = accelerometer_df.iloc[start_idx:end_idx, 2]
    z_accelerometer = accelerometer_df.iloc[start_idx:end_idx, 3]

    # Plot accelerometer data
    plt.figure(figsize=(12, 8))

    # Accelerometer X
    plt.subplot(3, 1, 1)
    plt.plot(time_interval, x_accelerometer, label="X")
    plt.title(f"Accelerometer X Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Accelerometer Y
    plt.subplot(3, 1, 2)
    plt.plot(time_interval, y_accelerometer, label="Y")
    plt.title(f"Accelerometer Y Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Accelerometer Z
    plt.subplot(3, 1, 3)
    plt.plot(time_interval, z_accelerometer, label="Z")
    plt.title(f"Accelerometer Z Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Extract gyroscope data
    x_gyroscope = gyroscope_df.iloc[start_idx:end_idx, 1]
    y_gyroscope = gyroscope_df.iloc[start_idx:end_idx, 2]
    z_gyroscope = gyroscope_df.iloc[start_idx:end_idx, 3]

    # Plot gyroscope data
    plt.figure(figsize=(12, 8))

    # Gyroscope X
    plt.subplot(3, 1, 1)
    plt.plot(time_interval, x_gyroscope, label="X")
    plt.title(f"Gyroscope X Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)

    # Gyroscope Y
    plt.subplot(3, 1, 2)
    plt.plot(time_interval, y_gyroscope, label="Y")
    plt.title(f"Gyroscope Y Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)

    # Gyroscope Z
    plt.subplot(3, 1, 3)
    plt.plot(time_interval, z_gyroscope, label="Z")
    plt.title(f"Gyroscope Z Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Extract linear accelerometer data
    x_linear_accelerometer = linear_accelerometer_df.iloc[start_idx:end_idx, 1]
    y_linear_accelerometer = linear_accelerometer_df.iloc[start_idx:end_idx, 2]
    z_linear_accelerometer = linear_accelerometer_df.iloc[start_idx:end_idx, 3]

    # Plot linear accelerometer data
    plt.figure(figsize=(12, 8))

    # Linear Accelerometer X
    plt.subplot(3, 1, 1)
    plt.plot(time_interval, x_linear_accelerometer, label="X")
    plt.title(f"Linear Accelerometer X Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Linear Accelerometer Y
    plt.subplot(3, 1, 2)
    plt.plot(time_interval, y_linear_accelerometer, label="Y")
    plt.title(f"Linear Accelerometer Y Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Linear Accelerometer Z
    plt.subplot(3, 1, 3)
    plt.plot(time_interval, z_linear_accelerometer, label="Z")
    plt.title(f"Linear Accelerometer Z Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Function to plot mean and standard deviation for a specified folder
def plot_mean_stdev_for_folder(folder_path, show_plots=True):
    if not show_plots:
        return

    folder_name = os.path.basename(folder_path)
    mode = get_mode_of_transportation(folder_name)
    data_files = ["Accelerometer.csv", "Gyroscope.csv", "Linear Accelerometer.csv"]

    for data_file in data_files:
        file_path = os.path.join(folder_path, data_file)
        if os.path.isfile(file_path):
            data_type = data_file.split('.')[0]
            df = pd.read_csv(file_path)
            mean_values = df.mean().drop("Time (s)", errors='ignore')
            stdev_values = df.std().drop("Time (s)", errors='ignore')

            plt.figure(figsize=(14, 6))

            plt.bar(mean_values.index, mean_values.values, yerr=stdev_values.values, capsize=5, color='b', alpha=0.6)
            plt.ylabel('Values')
            plt.xlabel('Axis')
            plt.title(f'Mean and Standard Deviation of {data_type} for {mode}')
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.tight_layout()
            plt.show()

# Example calls to the functions
plot_grouped_by_mode(show_plots=False, filtered=True)
plot_all_modes_by_measurement(show_plots=False, filtered=True)
provide_data_insights()

# Define the paths to the filtered CSV files
auto_1_path = os.path.join("Data_Filtered", "bus_1")
auto_2_path = os.path.join("Data_Filtered", "bus_2")
auto_horizontaal_path = os.path.join("Data_Arthur", "auto_horizontaal")

# plot_time_series(auto_1_path, "auto_1", interval_start=0, interval_end=30)
# plot_time_series(auto_2_path, "fiets_2", interval_start=0, interval_end=30)
#
# # Plot the time series data for auto_horizontaal
# plot_time_series(auto_horizontaal_path, "auto_horizontaal")

# Plot mean and standard deviation for a specified folder
plot_mean_stdev_for_folder(auto_1_path)
plot_mean_stdev_for_folder(auto_2_path)

