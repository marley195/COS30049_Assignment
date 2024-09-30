import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data import get_normalization_layer
import tensorflow as tf
from tensorflow.keras import layers

def regression_model(X_train, X_test, y_train, y_test):

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model using the training data
    model.fit(X_train, y_train)

    # Predict the target values for the test set
    y_test_pred = model.predict(X_test)

    # Calculate evalutation metrics : 
    mse = mean_squared_error(y_test, y_test_pred)
    r2_score = model.score(X_test, y_test)

    # Print the evaluation metrics :
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2_score)

    """ 
    Plot Actual vs Predicted values for Linear Regression in a scatter plot
    Blue dots represents the actual vs predicted AQI values
    Red line shows the perfect prediction (y=x) 
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    plt.xlabel('Actual AQI') # Label for x-axis
    plt.ylabel('Predicted AQI') # Label for y-axis
    plt.title('Actual vs Predicted AQI (Linear Regression)') # Plot title
    plt.show()

    return model

# Build model and load it with training data.
def classification_model(df, aqc_map, train_ds):
	# Normalise Data
	aqc_count = len(aqc_map)
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
	
	return model

def train_classification(model, train_ds, val_ds, epochs):
	model.fit(train_ds, epochs=epochs, validation_data=val_ds)
	
def use_classification(model, input_columns, aqc_map, sample):
	sample = {k: sample[k] for k in sample.keys() & input_columns}

	input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
	#predictions = reloaded_model.predict(input_dict)
	predictions = model.predict(input_dict)
	prob = tf.nn.sigmoid(predictions[0])

	raw_value = float(prob[0]) * (len(aqc_map) - 1)
	rounded_value = int(round(raw_value))
	
	return aqc_map[rounded_value]
