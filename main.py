import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import load_data, plot_data, preprocess_data
from model import regression_model

def main():

    file_name = "data/city_day.csv"
    
    data = load_data(file_name)

    X_train, X_test, y_train, y_test = preprocess_data(data)

    model = regression_model(X_train, X_test, y_train, y_test)


    plot_data(data)

if __name__ == "__main__":
    main()