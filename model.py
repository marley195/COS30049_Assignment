import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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