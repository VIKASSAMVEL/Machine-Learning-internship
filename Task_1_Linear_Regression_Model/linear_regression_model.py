import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load the dataset
try:
    df = pd.read_csv("Stock Prices Data Set.csv")
except FileNotFoundError:
    print("Error: 'Stock Prices Data Set.csv' not found. Make sure the file is in the same directory as the script.")
    exit()

print("Dataset loaded successfully. First 5 rows:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# 2. Preprocess the data
# For simplicity, we'll drop rows with any missing values.
# In a real-world scenario, more sophisticated imputation techniques might be used.
df.dropna(inplace=True)
print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# Assuming the last column is the target variable (e.g., 'Price' or 'SalePrice')
# And all other columns are features.
# You might need to adjust this based on your actual dataset's column names.
# For this example, let's assume the target column is named 'Price' or similar.
# If your dataset has a different target column name, please adjust 'target_column_name'.

# Define features (X) and target (y)
features = ['open', 'high', 'low']
target = 'close'

X = df[features]
y = df[target]

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Target variable identified as: '{target}'")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# 3. Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training complete.")

# 4. Interpret the model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# 5. Evaluate the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

print("\nLinear regression model built and evaluated successfully.")
