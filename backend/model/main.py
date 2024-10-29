from data import clean_data, regression_data, clean_dataset, prepare_dataset
from model import regression_model, classification_model, train_classification, use_classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import argparse
import tensorflow as tf

def plot_scatter(test_dataset, model):
    # Plot the regression line
        # Visualize true vs. predicted labels
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
    # Confusion Matrix for AQI Classification
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix using seaborn's heatmap for better visualization
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=aqib_map.values(), yticklabels=aqib_map.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for AQI Classification')
    plt.show()

def main(action):
    # Create variables for classification model + file name
    batch_size = 256
    epochs = 100
    file_name = "backend/data/city_day.csv"
    
    if action == "train":
        # Clean data for regression model
        data = clean_data(file_name)

        # Build and train regression model
        X_train, _, y_train, _ = regression_data(data)
        regression_model_instance = regression_model(X_train, y_train)
        
        # Save the regression model
        with open("regression_model.pkl", "wb") as file:
            pickle.dump(regression_model_instance, file)

        # Data pre-processing for classification model
        df, aqc_map = clean_dataset(file_name)
        train_ds, val_ds, test_ds = prepare_dataset(df, batch_size)
        
        # Build and train classification model
        class_model = classification_model(df, aqc_map, train_ds)
        train_classification(class_model, train_ds, val_ds, epochs)
        
        # Save the classification model
        class_model.save("backend/model/classification_model.keras")
        df['AQI_Bucket'] = df['AQI_Bucket'].astype('category')
        aqib_map = dict(enumerate(df['AQI_Bucket'].cat.categories))

        print("Training complete. Models saved.")

    elif action == "predict":
        # Load the regression and classification models
        with open("backend/model/regression_model.pkl", "rb") as file:
            regression_model_instance = pickle.load(file)
        class_model = tf.keras.models.load_model("backend/model/classification_model.keras")

        # Data pre-processing for classification model
        df, aqc_map = clean_dataset(file_name)
        _, _, test_dataset = prepare_dataset(df, batch_size)
        df['AQI_Bucket'] = df['AQI_Bucket'].astype('category')
        aqib_map = dict(enumerate(df['AQI_Bucket'].cat.categories))

        # Get true and predicted labels for the classification model
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_pred = np.argmax(class_model.predict(test_ds), axis=1)

        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, aqib_map)

        # Plot scatter plot
        plot_scatter(test_dataset, class_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict using AQI models.")
    parser.add_argument("action", choices=["train", "predict"], help="Specify whether to train or predict.")
    args = parser.parse_args()
    main(args.action)
