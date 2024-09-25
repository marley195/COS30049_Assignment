import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import process_data, regression_data
from model import regression_model

def main():

    file_name = "data/city_day.csv"
    
    data = process_data(file_name)

    X_train, X_test, y_train, y_test = regression_data(data)

    model = regression_model(X_train, X_test, y_train, y_test)

    
if __name__ == "__main__":
    main()