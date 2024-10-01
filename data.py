import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
#Function for regression data cleaning.
def clean_data(file_name):
    #loaded data with paramerter file_name
    data = pd.read_csv(file_name)
    #dropped columns that are not needed
    data_cleaned = data.drop(columns=['City', 'Date']).dropna(subset=['AQI', 'AQI_Bucket'])
    #converted AQI to numeric
    data_cleaned['AQI'] = pd.to_numeric(data_cleaned['AQI'], errors='coerce')
    #dropped rows with missing values
    data_cleaned.dropna(inplace=True)

    return data_cleaned

def regression_data(data):
    #dropped columns that are not needed
    X_aqi = data.drop(columns=['AQI', 'AQI_Bucket'])
    y_aqi = data['AQI']
    #split data into training and testing data
    X_train_aqi, X_test_aqi, y_train_aqi, y_test_aqi = train_test_split(X_aqi, y_aqi, test_size=0.2, random_state=42)
    #standardize the data
    scaler = StandardScaler()
    X_train_aqi = scaler.fit_transform(X_train_aqi)
    X_test_aqi = scaler.transform(X_test_aqi)

    return X_train_aqi, X_test_aqi, y_train_aqi, y_test_aqi

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    # Function to convert dataframe to dataset
    df = dataframe.copy()
    labels = df.pop('AQI_Bucket')
    df = {key: value.to_numpy()[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    # Shuffle the data
    if shuffle:
      ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)
    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

def clean_dataset(filename):
  #read file to dataframe
  df = pd.read_csv(filename)
  
  # Clean Data
  df = df.drop(columns=['City', 'Date', 'AQI'])
  df = df.dropna(axis=1, how='all')
  df = df.dropna(subset='AQI_Bucket')
  # Fill missing values with average (median) values.
  averages = df.median(numeric_only=True).to_dict()
  #df = df.fillna(averages)
  df = df.dropna()
  # Serialise Classification Data
  # Get Classification Name to Code Mapping
  df['AQI_Bucket'] = df['AQI_Bucket'].astype('category')
  aqc_map = dict(enumerate(df['AQI_Bucket'].cat.categories))
  # Apply Classification Codes
  df['AQI_Bucket'] = df['AQI_Bucket'].cat.codes
    
  return (df, aqc_map)

def prepare_dataset(df, batch_size):
	# Split Data
	train, val, test = np.split(df, [int(0.8*len(df)), int(0.9*len(df))])
	# Apply Batch Size
	train_ds = df_to_dataset(train, batch_size=batch_size)
	val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
	test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
	
	return (train_ds, val_ds, test_ds)