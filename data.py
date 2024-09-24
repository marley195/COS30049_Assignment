import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(file_name):
    # Load the data
    data = pd.read_csv(file_name)

    # Drop the rows with missing values
    data.dropna(inplace=True)

    # Drop the columns that are not required
    data.drop(['City', 'Date'], axis=1, inplace=True)

    return data

def preprocess_data(test_data):
    # Pre-process the data
    scaler = StandardScaler()
    numerical_columns = test_data.drop(columns=['AQI_Bucket'])
    
    # Separate features and target
    y = test_data['AQI'].values
    X = numerical_columns.drop(columns=['AQI'])  # Exclude target variable from features

    # Split the data before scaling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the scaler only on the training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
