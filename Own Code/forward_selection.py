import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_data = pd.read_csv('Sets/pca_time_freq/train_data_pca_time_freq.csv')
val_data = pd.read_csv('Sets/pca_time_freq/val_data_pca_time_freq.csv')
test_data = pd.read_csv('Sets/pca_time_freq/test_data_pca_time_freq.csv')

# Separate features and labels
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_val = val_data.drop(columns=['label'])
y_val = val_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Best hyperparameters found
best_params = {
    'bootstrap': False,
    'max_depth': 20,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 150
}

# Initialize variables for forward selection
selected_features = ["X_highest_freq.2", "Z_highest_freq.1", "Y_highest_freq", "Y_spectral_energy.2"]
remaining_features = list(X_train.columns)
best_accuracy = 0

# # Forward Selection process
# while remaining_features:
#     feature_to_add = None
#     for feature in remaining_features:
#         current_features = selected_features + [feature]
#         # Train the Random Forest classifier on the current set of features
#         rf_classifier = RandomForestClassifier(**best_params, random_state=42)
#         rf_classifier.fit(X_train[current_features], y_train)
#         # Predict the labels for the validation set
#         val_predicted_labels = rf_classifier.predict(X_val[current_features])
#         # Evaluate the accuracy of the model
#         accuracy = accuracy_score(y_val, val_predicted_labels)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             feature_to_add = feature
#
#     if feature_to_add is None:
#         break
#
#     selected_features.append(feature_to_add)
#     remaining_features.remove(feature_to_add)
#     print(f'Feature added: {feature_to_add}, Validation accuracy: {best_accuracy:.2f}')

print(f'Selected features: {selected_features}')

# Train final model on the selected features
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train[selected_features], y_train)

# Predict and evaluate the final model on the validation set
final_val_predictions = final_model.predict(X_val[selected_features])
final_val_accuracy = accuracy_score(y_val, final_val_predictions)
print(f'\nFinal Validation Accuracy: {final_val_accuracy:.2f}')
print("\nValidation Classification Report:")
print(classification_report(y_val, final_val_predictions))

# Predict and evaluate the final model on the test set
final_test_predictions = final_model.predict(X_test[selected_features])
final_test_accuracy = accuracy_score(y_test, final_test_predictions)
print(f'\nFinal Test Accuracy: {final_test_accuracy:.2f}')
print("\nTest Classification Report:")
print(classification_report(y_test, final_test_predictions))

# Plot the confusion matrix for the validation set
val_conf_matrix = confusion_matrix(y_val, final_val_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Validation Set Confusion Matrix')
plt.show()

# Plot the confusion matrix for the test set
test_conf_matrix = confusion_matrix(y_test, final_test_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Set Confusion Matrix')
plt.show()
