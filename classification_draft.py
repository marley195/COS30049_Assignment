import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras import layers

# Parameters

batch_size = 256

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

dataframe = dataframe.drop(columns=['City', 'Date', 'AQI'])
dataframe = dataframe.dropna(axis=1, how='all')
dataframe = dataframe.dropna(subset='AQI_Bucket')

# Fill missing values with median.
averages = dataframe.median(numeric_only=True).to_dict()
dataframe = dataframe.dropna()

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

history = model.fit(train_ds, epochs=50, validation_data=val_ds)
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

sample = sample1

sample.update({k: v for k, v in averages.items() if v})
sample = {k: sample[k] for k in sample.keys() & numeric_columns}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
#predictions = reloaded_model.predict(input_dict)
predictions = model.predict(input_dict)
prob = tf.nn.sigmoid(predictions[0])

raw_value = float(prob[0]) / (len(aqib_map) - 1)
rounded_value = int(round(raw_value))
print(f"Prediction: {aqib_map[rounded_value]}")



# Assuming you have the test labels and predictions:
y_true = np.concatenate([y for x, y in test_ds], axis=0)  # True labels from the test dataset
y_pred = np.argmax(model.predict(test_ds), axis=1)  # Predicted labels

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix using seaborn's heatmap for better visualization
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=aqib_map.values(), yticklabels=aqib_map.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for AQI Classification')
plt.show()

# Select a sample from the test dataset
for batch in test_ds.take(1):
    inputs, labels = batch

# Get predictions
predictions = model.predict(inputs)
predicted_classes = np.argmax(predictions, axis=1)

# Plot true vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(range(len(labels)), labels, label="True Labels", color="blue")
plt.scatter(range(len(predicted_classes)), predicted_classes, label="Predicted Labels", color="red", marker='x')
plt.title('True vs Predicted AQI Buckets')
plt.xlabel('Sample Index')
plt.ylabel('AQI Bucket')
plt.legend()
plt.show()