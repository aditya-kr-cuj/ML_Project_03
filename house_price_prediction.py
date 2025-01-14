# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = '/Users/adi/Adi_coading/Python3/ML_Project_3/Housing.csv'  # Update with your actual file path
housing_data = pd.read_csv(file_path)

# Encode categorical variables
categorical_columns = ['mainroad', 'guestroom', 'basement', 
                       'hotwaterheating', 'airconditioning', 'prefarea']
for col in categorical_columns:
    housing_data[col] = housing_data[col].map({'yes': 1, 'no': 0})

# One-hot encode 'furnishingstatus'
housing_data = pd.get_dummies(housing_data, columns=['furnishingstatus'], drop_first=True)

# Define features (X) and target variable (y)
X = housing_data.drop(columns=['price'])
y = housing_data['price']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Manual RMSE calculation

# Display results
print("Model Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the trained model
import joblib
joblib.dump(model, 'house_price_model.pkl')
print("\nModel saved as 'house_price_model.pkl'.")

import matplotlib.pyplot as plt

# Scatter plot: Actual vs Predicted Prices
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

errors = y_test - y_pred

# Plot Error Distribution
plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()

import tkinter as tk
from tkinter import messagebox

# Popup for Performance Metrics
def show_performance():
    performance_message = (
        f"Model Performance:\n"
        f"Mean Absolute Error (MAE): {mae}\n"
        f"Mean Squared Error (MSE): {mse}\n"
        f"Root Mean Squared Error (RMSE): {rmse}"
    )
    messagebox.showinfo("Performance Metrics", performance_message)

# Create GUI window
root = tk.Tk()
root.title("Model Metrics")

# Button to Show Metrics
btn = tk.Button(root, text="Show Model Performance", command=show_performance)
btn.pack(pady=20)

root.mainloop()
