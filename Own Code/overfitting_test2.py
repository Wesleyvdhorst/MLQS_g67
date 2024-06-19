import pandas as pd
from scipy.stats import f_oneway, chi2_contingency

def get_data(file_path):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(file_path)

    # Drop columns 'Time (s)', 'Time (s).1', 'Time (s).2' if they exist
    columns_to_drop = ['Time (s)', 'Time (s).1', 'Time (s).2']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df

def compare_sets_statistical_tests(train_df, test_df):
    results = []
    insignificant_count = 0  # Counter for insignificant p-values

    # Iterate over each feature
    for feature in train_df.columns:
        if train_df[feature].dtype == 'float64':  # Numerical feature
            train_data = train_df[feature].dropna().values
            test_data = test_df[feature].dropna().values

            # Perform ANOVA test
            f_stat, p_value = f_oneway(train_data, test_data)

            # Determine significance
            significance = 'Significant' if p_value < 0.05 else 'Not significant'
            if significance == 'Not significant':
                insignificant_count += 1

            # Store the results
            results.append({
                'Feature': feature,
                'Statistic': f"ANOVA F-statistic: {f_stat:.4f}",
                'p-value': f"{p_value:.4f}",
                'Significance': significance
            })

        elif train_df[feature].dtype == 'object':  # Categorical feature
            # Create contingency table
            train_counts = train_df[feature].value_counts()
            test_counts = test_df[feature].value_counts()

            # Combine counts into a single DataFrame
            combined_counts = pd.concat([train_counts, test_counts], axis=1,
                                        keys=['auto1', 'auto2']).fillna(0)

            # Perform chi-square test
            chi2_stat, p_value, _, _ = chi2_contingency(combined_counts)

            # Determine significance
            significance = 'Significant' if p_value < 0.05 else 'Not significant'
            if significance == 'Not significant':
                insignificant_count += 1

            # Store the results
            results.append({
                'Feature': feature,
                'Statistic': f"Chi-square statistic: {chi2_stat:.4f}",
                'p-value': f"{p_value:.4f}",
                'Significance': significance
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, insignificant_count

# File paths
file_paths = {
    'auto1': 'Data_Features/trein_1/Linear Accelerometer_pca_time_freq.csv',
    'auto2': 'Data_Features/trein_2/Linear Accelerometer_pca_time_freq.csv',
}

# Load data
train_df = get_data(file_paths['auto1'])
test_df = get_data(file_paths['auto2'])

# Perform statistical tests and get results DataFrame and insignificant count
results_df, insignificant_count = compare_sets_statistical_tests(train_df, test_df)

# Convert results DataFrame to LaTeX table
latex_table = results_df.to_latex(index=False, escape=False, column_format='lccc')

# Print LaTeX table (you can copy this output into your LaTeX document)
print(latex_table)

# Print the number of insignificant features
print(f"Number of features with insignificant p-values: {insignificant_count}/{len(results_df)}")
