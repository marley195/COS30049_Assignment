import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import joblib

from data.data import clean_data, regression_data, clean_dataset, prepare_dataset
from model import regression_model, classification_model, train_classification

def plot_scatter(test_dataset, model):
    """Plot the true vs predicted AQI Buckets."""
    for batch in test_dataset.take(1):
        inputs, labels = batch
    predictions = model.predict(inputs)
    predicted_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(labels)), labels, label="True Labels", color="blue")
    plt.scatter(range(len(predicted_classes)), predicted_classes, label="Predicted Labels", color="red", marker='x')
    plt.title('True vs Predicted AQI Buckets')
    plt.xlabel('Sample Index')
    plt.ylabel('AQI Bucket')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, aqib_map):
    """Plot the confusion matrix for AQI classification."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=aqib_map.values(), yticklabels=aqib_map.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for AQI Classification')
    plt.show()

def train_models(file_name, batch_size, epochs):
    """Train and save both regression and classification models."""
    # Prepare data for regression model
    data = clean_data(file_name)
    X_train, _, y_train, _ = regression_data(data)
    reg_model = regression_model(X_train, y_train)
    joblib.dump(reg_model, "regression_model.joblib")
    
    # Prepare data for classification model
    df, aqc_map = clean_dataset(file_name)
    train_ds, val_ds, _ = prepare_dataset(df, batch_size)
    class_model = classification_model(df, aqc_map, train_ds)
    train_classification(class_model, train_ds, val_ds, epochs)
    class_model.save("classification_model.keras")

    print("Training complete. Models saved.")

def predict(file_name, batch_size):
    """Load models, perform predictions, and plot results."""
    # Load models
    reg_model = joblib.load("regression_model.joblib")
    class_model = tf.keras.models.load_model("classification_model.keras")
    
    # Prepare test dataset for classification model
    df, _ = clean_dataset(file_name)
    _, _, test_ds = prepare_dataset(df, batch_size)
    df['AQI_Bucket'] = df['AQI_Bucket'].astype('category')
    aqib_map = dict(enumerate(df['AQI_Bucket'].cat.categories))

    # Get true and predicted labels for the classification model
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = np.argmax(class_model.predict(test_ds), axis=1)

    # Plot confusion matrix and scatter plot
    plot_confusion_matrix(y_true, y_pred, aqib_map)
    plot_scatter(test_ds, class_model)

def main(action, file_name="data/city_day.csv", batch_size=256, epochs=100):
    """Main function to either train or predict."""
    if action == "train":
        train_models(file_name, batch_size, epochs)
    elif action == "predict":
        predict(file_name, batch_size)
    else:
        print("Invalid action. Choose 'train' or 'predict'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict using AQI models.")
    parser.add_argument("action", choices=["train", "predict"], help="Specify whether to train or predict.")
    args = parser.parse_args()
    main(args.action)
