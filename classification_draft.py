import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras import layers

# Parameters

batch_size = 256
epochs = 100

# Functions

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('AQI_Bucket')
  df = {key: value.to_numpy()[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
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

# Load Data

csv_file = 'data/city_day.csv'
dataframe = pd.read_csv(csv_file)

# Clean Data

print(f"Loaded Rows: {len(dataframe)}")

dataframe = dataframe.drop(columns=['City', 'Date', 'AQI'])
dataframe = dataframe.dropna(axis=1, how='all')
dataframe = dataframe.dropna(subset='AQI_Bucket')

# Fill missing values with median.
averages = dataframe.median(numeric_only=True).to_dict()
dataframe = dataframe.fillna(averages)
#dataframe = dataframe.dropna()

print(f"Remaining Rows: {len(dataframe)}")

# Serialise Classification Data

# Get Classification Name to Code Mapping
dataframe['AQI_Bucket'] = dataframe['AQI_Bucket'].astype('category')
aqib_map = dict(enumerate(dataframe['AQI_Bucket'].cat.categories))

# Apply Classification Codes
dataframe['AQI_Bucket'] = dataframe['AQI_Bucket'].cat.codes

# Split Data

train, val, test = np.split(dataframe, [int(0.8*len(dataframe)), int(0.9*len(dataframe))])

# Apply Batch Size

train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Normalise Data

all_inputs = {}
encoded_features = []
numeric_columns = dataframe.select_dtypes(include='number').columns.tolist()
numeric_columns.remove('AQI_Bucket')

# Numerical features.
for header in numeric_columns:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs[header] = numeric_col
  encoded_features.append(encoded_numeric_col)

# Build Model
num_classes = len(aqib_map)
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(64, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(all_inputs, output)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"],
              run_eagerly=True)

# Train Model

model.fit(train_ds, epochs=epochs, validation_data=val_ds)
result = model.evaluate(test_ds, return_dict=True)

print(f"Model Training Result: {result}")

#model.save('aq_classifier.keras')
#reloaded_model = tf.keras.models.load_model('aq_classifier.keras')

sample1 = {
    'PM2.5': 51.93,
    'PM10': None,
    'NO': 13.71,
    'NO2': 17.87,
    'NOx': 31.64,
    'NH3': None,
    'CO': 13.71,
    'SO2': 15.33,
    'O3': 12.88,
    'Benzene': 6.32,
    'Toluene': 19.17,
    'Xylene': 10.57,
} # Expected: Poor

sample2 = {
    'PM2.5': 71.56,
    'PM10': None,
    'NO': 3.51,
    'NO2': 16.83,
    'NOx': 20.32,
    'NH3': None,
    'CO': 3.51,
    'SO2': 30.28,
    'O3': 57.16,
    'Benzene': 3.95,
    'Toluene': 8.01,
    'Xylene': 0.63
} # Expected: Moderate

sample3 = {
    'PM2.5': 128.64,
    'PM10': 202.77,
    'NO': 11.96,
    'NO2': 14.02,
    'NOx': 21.02,
    'NH3': 6.05,
    'CO': 0.66,
    'SO2': 10.06,
    'O3': 19.17,
    'Benzene': 2.99,
    'Toluene': 3.0,
    'Xylene': 2.0
} # Expected: Poor

sample4 = {
    'PM2.5': 100.72,
    'PM10': 152.72,
    'NO': 17.77,
    'NO2': 29.84,
    'NOx': 2.87,
    'NH3': 39.74,
    'CO': 1.06,
    'SO2': 4.98,
    'O3': 23.98,
    'Benzene': 1.92,
    'Toluene': 2.85,
    'Xylene': 0.1
} # Expected: Poor

sample = sample4

#sample.update({k: v for k, v in averages.items() if v})

# remove features that were completely empty
sample = {k: sample[k] for k in sample.keys() & numeric_columns}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
#predictions = reloaded_model.predict(input_dict)
predictions = model.predict(input_dict)
prob = tf.nn.sigmoid(predictions[0])

denormed_value = float(prob[0]) * (len(aqib_map) - 1)

print(denormed_value)
print(aqib_map)
rounded_value = int(round(denormed_value))
print(f"Prediction: {aqib_map[rounded_value]}")

#dropna
#Model Training Result: {'accuracy': 0.9855769276618958, 'loss': 0.057369641959667206}
#fillna
#Model Training Result: {'accuracy': 1.0, 'loss': 0.001120519358664751}