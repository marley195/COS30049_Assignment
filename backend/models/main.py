from data import clean_data, regression_data, clean_dataset, prepare_dataset
from model import regression_model, classification_model, train_classification, use_classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

def main():
    #create variables for classifcation model + file name
    batch_size = 256
    epochs = 100
    file_name = "backend/data/city_day.csv"

    # clean data for regression model
    data = clean_data(file_name)

    # build + Train regression model
    X_train, X_test, y_train, y_test = regression_data(data)
    reg_model = regression_model(X_train, X_test, y_train, y_test)
    with open("regression_model.pkl", "wb") as file:
        pickle.dump(reg_model, file)
    # data pre-processing for classification model
    df, aqc_map = clean_dataset(file_name)
    train_ds, val_ds, test_ds = prepare_dataset(df, batch_size)
    # Build Classification Model
    class_model = classification_model(df, aqc_map, train_ds)
    # Train Classification Model
    train_classification(class_model, train_ds, val_ds, epochs)
    class_model.save("classification_model.h5")
    df['AQI_Bucket'] = df['AQI_Bucket'].astype('category')
    aqib_map = dict(enumerate(df['AQI_Bucket'].cat.categories))

    y_true = np.concatenate([y for x, y in test_ds], axis=0)  # True labels from the test dataset
    y_pred = np.argmax(class_model.predict(test_ds), axis=1)  # Predicted labels

    # Confusion Matrix for AQI Classification
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix using seaborn's heatmap for better visualization
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=aqib_map.values(), yticklabels=aqib_map.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for AQI Classification')
    plt.show()


    # Get a batch of data
    for batch in test_ds.take(1):
        inputs, labels = batch
    # Get predictions
    predictions = class_model.predict(inputs)
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

if __name__ == "__main__":
    main()
