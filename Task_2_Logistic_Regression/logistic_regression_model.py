import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# Define file paths
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os

# Import SMOTE for handling imbalanced data
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: imblearn library not found. Please install it using 'pip install imbalanced-learn'")
    exit()

# Define file paths
train_file = "e:\\codveda\\Machine Learning Task List\\Task_2_Logistic_Regression\\Churn Prdiction Data\\churn-bigml-80.csv"
test_file = "e:\\codveda\\Machine Learning Task List\\Task_2_Logistic_Regression\\Churn Prdiction Data\\churn-bigml-20.csv"

# 1. Load the datasets
try:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
except FileNotFoundError:
    print("Error: Dataset files not found. Make sure 'churn-bigml-80.csv' and 'churn-bigml-20.csv' are in the 'Churn Prdiction Data' directory.")
    exit()

print("Training dataset loaded successfully. First 5 rows:")
print(train_df.head())
print("\nTesting dataset loaded successfully. First 5 rows:")
print(test_df.head())

print("\nTraining Dataset Info:")
train_df.info()
print("\nTesting Dataset Info:")
test_df.info()

# 2. Preprocess the data
# Convert 'International plan' and 'Voice mail plan' to numerical (Yes/No to 1/0)
train_df['International plan'] = train_df['International plan'].apply(lambda x: 1 if x == 'Yes' else 0)
train_df['Voice mail plan'] = train_df['Voice mail plan'].apply(lambda x: 1 if x == 'Yes' else 0)
test_df['International plan'] = test_df['International plan'].apply(lambda x: 1 if x == 'Yes' else 0)
test_df['Voice mail plan'] = test_df['Voice mail plan'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert 'Churn' to numerical (True/False to 1/0)
train_df['Churn'] = train_df['Churn'].astype(int)
test_df['Churn'] = test_df['Churn'].astype(int)

# One-hot encode 'State' feature
train_df = pd.get_dummies(train_df, columns=['State'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['State'], drop_first=True)

# Align columns - very important for consistent feature sets
# Get missing columns in test_df that are in train_df
missing_cols_in_test = set(train_df.columns) - set(test_df.columns)
for c in missing_cols_in_test:
    test_df[c] = 0
# Get missing columns in train_df that are in test_df
missing_cols_in_train = set(test_df.columns) - set(train_df.columns)
for c in missing_cols_in_train:
    train_df[c] = 0

# Ensure the order of columns is the same in both datasets
test_df = test_df[train_df.columns]

# Define features (X) and target (y)
X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

print(f"\nTraining features (X_train) shape: {X_train.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")

# Apply SMOTE to the training data
print("\nApplying SMOTE to balance the training data...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Resampled training data shape: {X_train_res.shape}")
print(f"Resampled training target distribution:\n{y_train_res.value_counts()}")

# 3. Hyperparameter Tuning with GridSearchCV
print("\nPerforming GridSearchCV for hyperparameter tuning...")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000] # Ensure convergence
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

print("\nModel training complete with best parameters.")

# 4. Interpret the model coefficients and odds ratios
print("\nModel Coefficients and Odds Ratios (from best model):")
coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': best_model.coef_[0]})
coefficients['Odds Ratio'] = np.exp(coefficients['Coefficient'])
print(coefficients.sort_values(by='Odds Ratio', ascending=False))
print(f"Intercept: {best_model.intercept_[0]:.4f}")

# 5. Evaluate the model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Optimized Model)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("e:\\codveda\\Machine Learning Task List\\Task_2_Logistic_Regression\\roc_curve_optimized.png")
print("\nROC curve saved as 'roc_curve_optimized.png'")

print("\nLogistic regression model built and evaluated successfully (Optimized).")



