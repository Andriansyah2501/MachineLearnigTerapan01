# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# --- Data Collection and Preprocessing ---

# Loading the diabetes dataset to a pandas DataFrame
# Make sure 'diabetes.csv' is in the same directory as your script,
# or provide the full path to the file.
diabetes_dataset = pd.read_csv('diabetes.csv')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(diabetes_dataset.head())

# Get the shape of the dataset (number of rows and columns)
print("\nShape of the dataset (rows, columns):")
print(diabetes_dataset.shape)

# Get statistical measures of the data
print("\nStatistical measures of the dataset:")
print(diabetes_dataset.describe())

# Count the number of non-diabetic (0) and diabetic (1) cases
print("\nNumber of non-diabetic (0) and diabetic (1) cases:")
print(diabetes_dataset['Outcome'].value_counts())

# Group the data by 'Outcome' and calculate the mean for each feature
print("\nMean of features grouped by Outcome:")
print(diabetes_dataset.groupby('Outcome').mean())

# Separate features (X) and target (Y)
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data

# --- Train-Test Split ---

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# stratify=Y ensures that the proportion of 'Outcome' classes is the same in both train and test sets
# random_state=2 ensures reproducibility of the split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print("\nShape of X_train, X_test, Y_train, Y_test:")
print(X.shape, X_train.shape, X_test.shape)

# --- Model Training (Support Vector Machine Classifier) ---

# Initialize the SVM classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Train the classifier on the training data
classifier.fit(X_train, Y_train)

# --- Model Evaluation ---

# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("\nAccuracy score on the training data:", training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score on the test data:", test_data_accuracy)

# --- Making a Predictive System ---

# Example input data (replace with your desired values)
# This example uses the first row of the original dataset for demonstration
# You should provide your own new data here.
input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50) # Example from the dataset

# Change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)
print("\nStandardized input data:", std_data)

# Make a prediction
prediction = classifier.predict(std_data)
print("\nPrediction (0 = Non-Diabetic, 1 = Diabetic):", prediction[0])

if (prediction[0] == 0):
    print("The person is not diabetic.")
else:
    print("The person is diabetic.")
