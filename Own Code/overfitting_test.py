import pandas as pd
from scipy.stats import f_oneway, chi2_contingency

def get_data(file_path):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(file_path)

    # Drop columns 'Time (s)', 'Time (s).1', 'Time (s).2' if they exist
    columns_to_drop = ['Time (s)', 'Time (s).1', 'Time (s).2']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df

def compare_sets_statistical_tests(train_df, test_df, val_df):
    results = []
    non_significant_count = {label: 0 for label in range(8)}  # Initialize count for each label
    total_tests = 0

    # Group dataframes by 'label' column
    train_groups = train_df.groupby('label')
    test_groups = test_df.groupby('label')
    val_groups = val_df.groupby('label')

    # Iterate over each label group
    for label in range(8):  # Assuming labels are from 0 to 7
        # Get data for this label from each set
        train_data = train_groups.get_group(label) if label in train_groups.groups else pd.DataFrame()
        test_data = test_groups.get_group(label) if label in test_groups.groups else pd.DataFrame()
        val_data = val_groups.get_group(label) if label in val_groups.groups else pd.DataFrame()

        # Iterate over each feature
        for feature in train_df.columns:
            if train_df[feature].dtype == 'float64':  # Numerical feature
                train_feature_data = train_data[feature].dropna().values
                test_feature_data = test_data[feature].dropna().values
                val_feature_data = val_data[feature].dropna().values

                # Perform ANOVA test
                f_stat, p_value = f_oneway(train_feature_data, test_feature_data, val_feature_data)
                total_tests += 1

                # Store the results
                results.append({
                    'Feature': feature,
                    'Label': label,
                    'Statistic': f"ANOVA F-statistic: {f_stat:.4f}",
                    'p-value': f"{p_value:.4f}",
                    'Significance': 'Significant' if p_value < 0.05 else 'Not significant'
                })

                # Count non-significant p-values per label
                if p_value >= 0.05:
                    non_significant_count[label] += 1

            elif train_df[feature].dtype == 'object':  # Categorical feature
                # Create contingency table
                train_counts = train_data[feature].value_counts()
                test_counts = test_data[feature].value_counts()
                val_counts = val_data[feature].value_counts()

                # Combine counts into a single DataFrame
                combined_counts = pd.concat([train_counts, test_counts, val_counts], axis=1,
                                            keys=['train', 'test', 'val']).fillna(0)

                # Perform chi-square test
                chi2_stat, p_value, _, _ = chi2_contingency(combined_counts)
                total_tests += 1

                # Store the results
                results.append({
                    'Feature': feature,
                    'Label': label,
                    'Statistic': f"Chi-square statistic: {chi2_stat:.4f}",
                    'p-value': f"{p_value:.4f}",
                    'Significance': 'Significant' if p_value < 0.05 else 'Not significant'
                })

                # Count non-significant p-values per label
                if p_value >= 0.05:
                    non_significant_count[label] += 1

    # Calculate percentage of non-significant p-values per label
    non_significant_percentage = {label: (count / len(train_df.columns)) * 100 for label, count in non_significant_count.items()}

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Convert DataFrame to LaTeX table format
    latex_table = results_df.to_latex(index=False, escape=False)

    return latex_table, non_significant_percentage

# File paths
file_paths = {
    'train': 'Sets2/pca_time_freq/train_data_pca_time_freq.csv',
    'test': 'Sets2/pca_time_freq/test_data_pca_time_freq.csv',
    'val': 'Sets2/pca_time_freq/val_data_pca_time_freq.csv'
}

# Load data
train_df = get_data(file_paths['train'])
test_df = get_data(file_paths['test'])
val_df = get_data(file_paths['val'])

# Generate LaTeX table and get non-significant percentage per label
latex_table, non_significant_percentage = compare_sets_statistical_tests(train_df, test_df, val_df)

# Print LaTeX table (you can copy this output into your LaTeX document)
print(latex_table)

# Print percentage of non-significant p-values per label
print("Percentage of non-significant p-values per label:")
for label, percentage in non_significant_percentage.items():
    print(f"Label {label}: {percentage:.2f}%")
