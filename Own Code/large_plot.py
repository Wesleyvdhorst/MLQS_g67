import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_stats(df):
    mean_values = df.mean()
    stdev_values = df.std()
    return mean_values, stdev_values


def process_csv_files(folder_path):
    stats = {}
    for file_name in ["Accelerometer.csv", "Gyroscope.csv", "Linear Accelerometer.csv"]:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = df.drop(df.columns[0], axis=1)  # Drop the first column (Time)
            mean_values, stdev_values = calculate_stats(df)
            stats[file_name] = (mean_values, stdev_values)
    return stats


def plot_stats(stats_by_folder, measurement, filtered=True):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f'Mean and Standard Deviation of {"Filtered" if filtered else "Original"} {measurement}')

    axes_labels = ["X", "Y", "Z"]
    subfolder_names = list(stats_by_folder.keys())
    bar_width = 0.2

    for i, axis in enumerate(axes_labels):
        index = np.arange(len(subfolder_names))

        mean_values_list = []
        stdev_values_list = []
        for subfolder in subfolder_names:
            mean_values, stdev_values = stats_by_folder[subfolder][measurement]
            mean_values_list.append(mean_values.iloc[i])
            stdev_values_list.append(stdev_values.iloc[i])

        axs[i].bar(index, mean_values_list, yerr=stdev_values_list, capsize=5, label=axis, width=bar_width, alpha=0.6)
        axs[i].set_xlabel('Subfolder')
        axs[i].set_ylabel(f'{axis} Acceleration (m/s^2)')
        axs[i].set_xticks(index)
        axs[i].set_xticklabels(subfolder_names, rotation=45)
        axs[i].grid(True)
        axs[i].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def process_data_filtered_folder(data_filtered_path, limit=3):
    stats_by_folder = {}
    count = 0

    for subfolder in os.listdir(data_filtered_path):
        if count >= limit:
            break
        subfolder_path = os.path.join(data_filtered_path, subfolder)
        if os.path.isdir(subfolder_path):
            stats_by_folder[subfolder] = process_csv_files(subfolder_path)
            count += 1

    return stats_by_folder


def visualize_data_filtered(data_filtered_path, show_plots=True, filtered=True):
    if not show_plots:
        return

    stats_by_folder = process_data_filtered_folder(data_filtered_path, limit=20)

    for measurement in ["Accelerometer.csv", "Gyroscope.csv", "Linear Accelerometer.csv"]:
        plot_stats(stats_by_folder, measurement, filtered)


# Example usage
data_filtered_path = "Data_Filtered"
visualize_data_filtered(data_filtered_path, show_plots=True, filtered=True)
