import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def process_data(file_name):

    #loaded data with paramerter file_name
    data = pd.read_csv(file_name)
    data_cleaned = data.drop(columns=['City', 'Date']).dropna(subset=['AQI', 'AQI_Bucket'])

    data_cleaned['AQI'] = pd.to_numeric(data_cleaned['AQI'], errors='coerce')
    data_cleaned.dropna(inplace=True)

    return data_cleaned

def regression_data(data):

    X_aqi = data.drop(columns=['AQI', 'AQI_Bucket'])
    y_aqi = data['AQI']

    X_train_aqi, X_test_aqi, y_train_aqi, y_test_aqi = train_test_split(X_aqi, y_aqi, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_aqi = scaler.fit_transform(X_train_aqi)
    X_test_aqi = scaler.transform(X_test_aqi)

    return X_train_aqi, X_test_aqi, y_train_aqi, y_test_aqi

def classifcation_data(data, scaler):

    X_bucket = data.drop(columns=['AQI', 'AQI_Bucket'])
    y_bucket = LabelEncoder().fit_transform(data['AQI_Bucket'])
    
    X_train_bucket, X_test_bucket, y_train_bucket, y_test_bucket = train_test_split(X_bucket, y_bucket, test_size=0.2, random_state=42)
    
    X_train_bucket = scaler.fit_transform(X_train_bucket)
    X_test_bucket = scaler.transform(X_test_bucket)

    return X_train_bucket, X_test_bucket, y_train_bucket, y_test_bucket