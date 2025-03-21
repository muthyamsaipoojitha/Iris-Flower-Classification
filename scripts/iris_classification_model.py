# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score

# Step 1: Load the dataset from the specified path with a different encoding
file_path = r'C:\Users\saira\Downloads\Iris-Flower-Classification\data\IRIS.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Print the first few rows of the dataset to check if it's loaded correctly
print("Dataset Loaded Successfully")
print(data.head())

# Step 2: Data Preprocessing
# Handle missing values (example: fill with mean or mode)
data['species'] = data['species'].fillna(data['species'].mode()[0])  # Replace missing species with the mode

# Feature Engineering (if needed) and data preparation
X = data.drop('species', axis=1)  # Features (sepal length, sepal width, petal length, petal width)
y = data['species']  # Target variable (flower species: Setosa, Versicolor, Virginica)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to improve performance of certain algorithms like KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Selection (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Predict using the trained model
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Cross-validation to check model performance across different splits
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean Cross-validation score: {cv_scores.mean()}")
