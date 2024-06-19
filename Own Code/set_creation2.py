import os
import pandas as pd

# Define the path to the filtered data folder
data_folder = "Data_Filtered"  # Update this to the path where your filtered data is stored

# Define a mapping from folder names to integer labels
label_mapping = {
    'auto': 0,
    'bus': 1,
    'fiets': 2,
    'lopen': 3,
    'metro': 4,
    'paard': 5,
    'tram': 6,
    'trein': 7
}

# Initialize a list to hold the combined data
combined_data_list = []

# Iterate over each subfolder in the data_filtered folder
for subfolder in os.listdir(data_folder):
    subfolder_path = os.path.join(data_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Read the accelerometer, gyroscope, and linear accelerometer CSV files
        accelerometer_file = os.path.join(subfolder_path, 'Accelerometer.csv')
        gyroscope_file = os.path.join(subfolder_path, 'Gyroscope.csv')
        linear_accelerometer_file = os.path.join(subfolder_path, 'Linear Accelerometer.csv')

        if os.path.exists(accelerometer_file) and os.path.exists(gyroscope_file) and os.path.exists(
                linear_accelerometer_file):
            acc_data = pd.read_csv(accelerometer_file)
            gyro_data = pd.read_csv(gyroscope_file)
            lin_acc_data = pd.read_csv(linear_accelerometer_file)

            # Reset index to align them properly for merging
            acc_data.reset_index(drop=True, inplace=True)
            gyro_data.reset_index(drop=True, inplace=True)
            lin_acc_data.reset_index(drop=True, inplace=True)

            merged_data = pd.concat([acc_data, gyro_data, lin_acc_data], axis=1)

            # Determine the label based on the folder name
            label = next((label_mapping[key] for key in label_mapping if key in subfolder), None)

            if label is not None:
                # Add the label column based on the folder name
                merged_data['label'] = label

            # Append the merged data to the list
            combined_data_list.append(merged_data)

# Concatenate all the combined data into a single DataFrame
combined_data = pd.concat(combined_data_list, ignore_index=True)

# Remove columns containing "Time" in their names
time_columns = [col for col in combined_data.columns if 'Time' in col]
combined_data = combined_data.drop(columns=time_columns)

# Define percentages for training, validation, and test
train_percent = 0.65
val_percent = 0.10
test_percent = 0.25

# Initialize empty lists to store data splits
train_data_list = []
val_data_list = []
test_data_list = []

# Group by 'label' and split each group
for label, group_data in combined_data.groupby('label'):
    # Sort the group data by index
    group_data = group_data.sort_index()

    # Calculate lengths for each split
    total_length = len(group_data)
    train_length = int(train_percent * total_length)
    val_length = int(val_percent * total_length)

    # Split the group data into training, validation, and test sets
    train_data = group_data.iloc[:train_length]
    val_data = group_data.iloc[train_length:train_length + val_length]
    test_data = group_data.iloc[train_length + val_length:]

    # Append to lists
    train_data_list.append(train_data)
    val_data_list.append(val_data)
    test_data_list.append(test_data)

# Concatenate all splits into final DataFrames
train_data = pd.concat(train_data_list, ignore_index=True)
val_data = pd.concat(val_data_list, ignore_index=True)
test_data = pd.concat(test_data_list, ignore_index=True)

# # Print lengths of each set to verify percentages
# print(f"Total combined data length: {len(combined_data)}")
# print(f"Training data length: {len(train_data)}, Percentage: {len(train_data) / len(combined_data) * 100:.2f}%")
# print(f"Validation data length: {len(val_data)}, Percentage: {len(val_data) / len(combined_data) * 100:.2f}%")
# print(f"Test data length: {len(test_data)}, Percentage: {len(test_data) / len(combined_data) * 100:.2f}%")
#
# # Optionally, print the first few rows of each set to verify
# print("\nTraining data:")
# print(train_data)
# print("\nValidation data:")
# print(val_data)
# print("\nTest data:")
# print(test_data)

# Save the datasets to CSV files
train_data.to_csv('Sets2/org/train_data_org.csv', index=False)
val_data.to_csv('Sets2/org/val_data_org.csv', index=False)
test_data.to_csv('Sets2/org/test_data_org.csv', index=False)

# Verify the split
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")