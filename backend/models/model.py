import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers
from data import get_normalization_layer
import numpy as np

# Regression Model
def regression_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set and evaluate
    y_test_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2_score = model.score(X_test, y_test)

    # Display evaluation metrics
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2_score)

    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI (Linear Regression)')
    plt.show()

    return model

# Classification Model
def classification_model(df, aqc_map, train_ds):
    aqc_count = len(aqc_map)
    all_inputs, encoded_features = {}, []
    input_columns = df.select_dtypes(include='number').columns.tolist()
    input_columns.remove('AQI_Bucket')

    # Normalize numerical features
    for header in input_columns:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs[header] = numeric_col
        encoded_features.append(encoded_numeric_col)

    # Build the classification model
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = layers.Dense(64, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(aqc_count, activation="softmax")(x)

    model = tf.keras.Model(inputs=all_inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"],
                  run_eagerly=True)

    return model

# Train Classification Model
def train_classification(model, train_ds, val_ds, epochs):
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# Use Classification Model for Prediction
def use_classification(model, input_columns, aqc_map, sample):
    sample = {k: sample[k] for k in sample.keys() & input_columns}
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}

    predictions = model.predict(input_dict)
    prob = tf.nn.sigmoid(predictions[0])

    raw_value = float(prob[0]) * (len(aqc_map) - 1)
    rounded_value = int(round(raw_value))

    return aqc_map[rounded_value]

# Function to predict AQI for a single sample using the regression model
def predict_aqi(regression_model, sample):
    # Convert the sample to the format expected by the model
    # Assuming sample is a dictionary, convert it to a numpy array with shape (1, -1)
    input_array = np.array([list(sample.values())])

    # Use the regression model to predict AQI
    aqi_prediction = regression_model.predict(input_array)

    return aqi_prediction[0]  # Return the predicted AQI value