import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers # type: ignore

def clean_data(file_name):
    """
    Loads and cleans data for regression.
    - Drops unnecessary columns.
    - Converts AQI to numeric.
    - Removes rows with missing AQI and AQI_Bucket values.
    """
    data = pd.read_csv(file_name)
    data_cleaned = data.drop(columns=['City', 'Date']).dropna(subset=['AQI', 'AQI_Bucket'])
    data_cleaned['AQI'] = pd.to_numeric(data_cleaned['AQI'], errors='coerce')
    data_cleaned.dropna(inplace=True)
    return data_cleaned

def regression_data(data):
    """
    Prepares data for regression model.
    - Drops AQI and AQI_Bucket columns to get features (X).
    - Uses AQI as the target (y).
    - Splits data into training and testing sets.
    - Standardizes the data using StandardScaler.
    """
    X_aqi = data.drop(columns=['AQI', 'AQI_Bucket'])
    y_aqi = data['AQI']
    X_train_aqi, X_test_aqi, y_train_aqi, y_test_aqi = train_test_split(X_aqi, y_aqi, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_aqi = scaler.fit_transform(X_train_aqi)
    X_test_aqi = scaler.transform(X_test_aqi)
    
    return X_train_aqi, X_test_aqi, y_train_aqi, y_test_aqi

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """
    Converts a DataFrame to a TensorFlow dataset for classification.
    - Extracts labels from the 'AQI_Bucket' column.
    - Creates a dictionary of features with each feature as a separate tensor.
    """
    df = dataframe.copy()
    labels = df.pop('AQI_Bucket')
    df = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size).prefetch(batch_size)
    return ds

def get_normalization_layer(name, dataset):
    """
    Creates a normalization layer for a specified feature.
    - Learns the statistics from the dataset and adapts the layer accordingly.
    """
    normalizer = layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)
    return normalizer

def clean_dataset(filename):
    """
    Loads and cleans data for classification.
    - Drops unnecessary columns.
    - Fills missing numeric values with median.
    - Encodes 'AQI_Bucket' as categorical codes.
    """
    df = pd.read_csv(filename)
    df = df.drop(columns=['City', 'Date', 'AQI']).dropna(axis=1, how='all').dropna(subset=['AQI_Bucket'])
    df.fillna(df.median(numeric_only=True), inplace=True)
    df['AQI_Bucket'] = df['AQI_Bucket'].astype('category')
    aqc_map = dict(enumerate(df['AQI_Bucket'].cat.categories))
    df['AQI_Bucket'] = df['AQI_Bucket'].cat.codes
    return df, aqc_map

def prepare_dataset(df, batch_size):
    """
    Splits data into train, validation, and test sets.
    - Converts each split into a TensorFlow dataset with batching.
    """
    train, val, test = np.split(df, [int(0.8*len(df)), int(0.9*len(df))])
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    return train_ds, val_ds, test_ds
