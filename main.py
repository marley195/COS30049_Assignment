import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import load_data, plot_data, preprocess_data

def main():

    file_name = "data/city_day.csv"
    
    data = load_data(file_name)

    X_train, X_test, y_train, y_test = preprocess_data(data)

    plot_data(data)

if __name__ == "__main__":
    main()