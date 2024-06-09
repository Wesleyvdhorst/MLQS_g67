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
            mean_values = mean_values_by_mode[mode][data_type]
            stdev_values = stdev_values_by_mode[mode][data_type]

            plt.figure(figsize=(14, 6))

            plt.bar(mean_values.index, mean_values.values, yerr=stdev_values.values, capsize=5, color='b', alpha=0.6)
            plt.ylabel('Acceleration (m/s^2)')
            plt.xlabel('Data Type')
            plt.title(f'Mean and Standard Deviation of {"Filtered" if filtered else "Original"} {data_type} for {mode}')
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.tight_layout()
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

# Example calls to the functions
plot_grouped_by_mode(show_plots=False, filtered=True)
provide_data_insights()

# Define the paths to the filtered CSV files
auto_1_path = os.path.join("Data_Filtered", "auto_1")
auto_horizontaal_path = os.path.join("Data_Arthur", "auto_horizontaal")


# Function to plot the time series data
def plot_time_series(folder_path, label):
    # Load the CSV files for Accelerometer, Gyroscope, and Linear Accelerometer
    accelerometer_df = pd.read_csv(os.path.join(folder_path, "Accelerometer.csv"))
    gyroscope_df = pd.read_csv(os.path.join(folder_path, "Gyroscope.csv"))
    linear_accelerometer_df = pd.read_csv(os.path.join(folder_path, "Linear Accelerometer.csv"))

    # Extract data from the dataframes
    time = accelerometer_df.iloc[:, 0]

    # Extract accelerometer data
    x_accelerometer = accelerometer_df.iloc[:, 1]
    y_accelerometer = accelerometer_df.iloc[:, 2]
    z_accelerometer = accelerometer_df.iloc[:, 3]

    # Plot accelerometer data
    plt.figure(figsize=(12, 8))

    # Accelerometer X
    plt.subplot(3, 1, 1)
    plt.plot(time, x_accelerometer, label="X")
    plt.title(f"Accelerometer X Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Accelerometer Y
    plt.subplot(3, 1, 2)
    plt.plot(time, y_accelerometer, label="Y")
    plt.title(f"Accelerometer Y Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Accelerometer Z
    plt.subplot(3, 1, 3)
    plt.plot(time, z_accelerometer, label="Z")
    plt.title(f"Accelerometer Z Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Extract gyroscope data
    x_gyroscope = gyroscope_df.iloc[:, 1]
    y_gyroscope = gyroscope_df.iloc[:, 2]
    z_gyroscope = gyroscope_df.iloc[:, 3]

    # Plot gyroscope data
    plt.figure(figsize=(12, 8))

    # Gyroscope X
    plt.subplot(3, 1, 1)
    plt.plot(time, x_gyroscope, label="X")
    plt.title(f"Gyroscope X Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)

    # Gyroscope Y
    plt.subplot(3, 1, 2)
    plt.plot(time, y_gyroscope, label="Y")
    plt.title(f"Gyroscope Y Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)

    # Gyroscope Z
    plt.subplot(3, 1, 3)
    plt.plot(time, z_gyroscope, label="Z")
    plt.title(f"Gyroscope Z Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Extract linear accelerometer data
    x_linear_accelerometer = linear_accelerometer_df.iloc[:, 1]
    y_linear_accelerometer = linear_accelerometer_df.iloc[:, 2]
    z_linear_accelerometer = linear_accelerometer_df.iloc[:, 3]

    # Plot linear accelerometer data
    plt.figure(figsize=(12, 8))

    # Linear Accelerometer X
    plt.subplot(3, 1, 1)
    plt.plot(time, x_linear_accelerometer, label="X")
    plt.title(f"Linear Accelerometer X Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Linear Accelerometer Y
    plt.subplot(3, 1, 2)
    plt.plot(time, y_linear_accelerometer, label="Y")
    plt.title(f"Linear Accelerometer Y Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    # Linear Accelerometer Z
    plt.subplot(3, 1, 3)
    plt.plot(time, z_linear_accelerometer, label="Z")
    plt.title(f"Linear Accelerometer Z Time Series Data for {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


#
# # Plot the time series data for auto_1
# plot_time_series(auto_1_path, "auto_1")
#
# # Plot the time series data for auto_horizontaal
# plot_time_series(auto_horizontaal_path, "auto_horizontaal")