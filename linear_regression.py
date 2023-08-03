import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("datasets/advertising.csv")
df.shape

X = df[["TV"]]
Y = df[["sales"]]

reg_model =LinearRegression().fit(X, Y)

# (b - bias)
reg_model.intercept_[0]

# (w1)
reg_model.coef_[0][0]

# prediction

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# Visualisation of Model

graph_of_model = sns.regplot(x=X, y=Y, scatter_kws={"color": "b", "s": 9}, ci=False, color="r")

graph_of_model.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
graph_of_model.set_ylabel("Satış Sayısı")
graph_of_model.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

# prediction success

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(Y, y_pred)

Y.mean()
Y.std()

# RMSE
np.sqrt(mean_squared_error(Y, y_pred))

# MAE
mean_absolute_error(Y, y_pred)

# R-squared
reg_model.score(X, Y)


### Multiple Linear Regression

df = pd.read_csv("datasets/advertising.csv")
X = df.drop("sales", axis=1)
Y = df[["sales"]]

# model

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

Y_test.shape
Y_train.shape

reg_model = LinearRegression().fit(X_train, Y_train)

reg_model.intercept_
reg_model.coef_

# prediction

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

# Evaluating Prediction Success

# Train RMSE
Y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(Y_train, Y_pred))
# 1.73

# TRAIN RSQUARED
reg_model.score(X_train, Y_train)

# Test RMSE
Y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(Y_test, Y_pred))
# 1.41

# Test RSQUARED
reg_model.score(X_test, Y_test)


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 Y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 Y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71


### Simple Linear Regression with Gradient Descent from Scratch

def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

