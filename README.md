# Iris Flower Classification

This project aims to classify Iris flowers into three species based on measurements of sepal length, sepal width, petal length, and petal width using machine learning algorithms.

## Approach:
1. **Data Preprocessing**:
   - Missing values are handled (though the Iris dataset has no missing values).
   - Features are standardized using `StandardScaler`.

2. **Modeling**:
   - A **RandomForestClassifier** is used for classification, but other models like **KNN** and **Logistic Regression** can also be tried.

3. **Evaluation**:
   - **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Cross-validation** are used to evaluate the model.

## Requirements:
- Python 3.x
- pandas
- numpy
- scikit-learn

## Installation:
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
