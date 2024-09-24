from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def regression_model(X_train, X_test, y_train, y_test):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    print("Mean Squared Error:", mse)

    score = model.score(X_test, y_test)
    print("R^2 Score:", score)

    return model