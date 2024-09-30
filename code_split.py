import numpy as np
import pandas as pd
import tensorflow as tf

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

def clean_dataset(df):
	# Clean Data

	df = df.drop(columns=['City', 'Date', 'AQI'])
	df = df.dropna(axis=1, how='all')
	df = df.dropna(subset='AQI_Bucket')

	# Fill missing values with average (median) values.
	averages = df.median(numeric_only=True).to_dict()
	df = df.fillna(averages)
	#df = df.dropna()
	
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

# Build model and load it with training data.
def build_model(df, aqc_count):
	# Normalise Data
	
	all_inputs = {}
	encoded_features = []
	input_columns = df.select_dtypes(include='number').columns.tolist()
	input_columns.remove('AQI_Bucket')
	
	# Numerical features.
	for header in input_columns:
		numeric_col = tf.keras.Input(shape=(1,), name=header)
		normalization_layer = get_normalization_layer(header, train_ds)
		encoded_numeric_col = normalization_layer(numeric_col)
		all_inputs[header] = numeric_col
		encoded_features.append(encoded_numeric_col)

	# Build Model
	
	#num_classes = len(aqc_map)
	all_features = tf.keras.layers.concatenate(encoded_features)
	x = tf.keras.layers.Dense(64, activation="relu")(all_features)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(32, activation="relu")(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	#output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
	output = tf.keras.layers.Dense(aqc_count, activation="softmax")(x)

	model = tf.keras.Model(all_inputs, output)

	model.compile(optimizer='adam',
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=["accuracy"],
				run_eagerly=True)
	
	return (model, input_columns)

def train_model(model, train_ds, val_ds, epochs):
	model.fit(train_ds, epochs=epochs, validation_data=val_ds)

def use_model(model, input_columns, aqc_map, sample):
	sample = {k: sample[k] for k in sample.keys() & input_columns}

	input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
	#predictions = reloaded_model.predict(input_dict)
	predictions = model.predict(input_dict)
	prob = tf.nn.sigmoid(predictions[0])

	raw_value = float(prob[0]) * (len(aqc_map) - 1)
	rounded_value = int(round(raw_value))
	
	return aqc_map[rounded_value]

# Load Data

csv_file = 'data/city_day.csv'
df = pd.read_csv(csv_file)

df, aqc_map = clean_dataset(df)
train_ds, val_ds, test_ds = prepare_dataset(df, batch_size)
model, input_columns = build_model(df, len(aqc_map))
train_model(model, train_ds, val_ds, epochs)

sample = {
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

print(use_model(model, input_columns, aqc_map, sample))