'''
This demo is meant to show Linear Regression in action (the "line of best fit" concept)

Essentially, this demo:

1. Loads numeric data from both .npy files (Xdata.npy, yData.npy)
2. Splits that data into a training and test set
3. Trains a Linear Regression model using scikit-learn
4. Plots the fitted line vs the data points,
5. Evaluates model performance using Mean Squared Error (MSE)

This demo is a real implementation of Linear Regression from 3 - Supervised Learning: Model Overfitting, Regression, and Gradient Descent
'''


import numpy as np # handles arrays and numerical operations
import matplotlib.pyplot as plt # for plotting data and fitted lines
from sklearn.model_selection import train_test_split # splits dataset into training/testing parts
from sklearn import linear_model # module containing LinearRegression class
from sklearn.metrics import mean_squared_error # metric to measure model performance

X = np.load(r"Xdata.npy")
y = np.load(r"Ydata.npy")
# loads arrays from .npy files
# `x` -> independent varaibles (features)
# `y` -> dependent variables (target/output)

plt.scatter(X,y, color = 'black')
plt.xlabel('X')
plt.ylabel('y')
# creates a scatter plot of data points (raw data)
# points shown in black represent actual observed samples
# the `xlabel` and `ylabel` label the axes
# this is the data visualization step, helps see whether the relation looks linear or not

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)
# split the dataset into training and test set:
# training dataset -> 70%
# test dataset -> 30%

# Create a linear regression object
reg = linear_model.LinearRegression()
# create a linear regression object, this creates an instance of scikit-learn's `LinearRegression` model
# internally, it willl learn weights (slope + intercept) using least squares

# Train the model
reg.fit(xTrain, yTrain)
# train the model
# this fits the regression line to the training data, computes w0 (intercept) and w1 (slope) to minimize squared errors

# Plot the linear fit
yPred = reg.predict(xTrain)
# uses the trained model to predict the training data outputs

plt.scatter(xTrain, yPred, color = 'r')
# plots the fitted values (in red) on top of the black training points

plt.show()
# this shows how well the red regression line follows the actual points
# the red line approximates the trend of the black dots

yPredTest = reg.predict(xTest)
# model evaluation, apply the model on the test dataset
# predicts outputs for unseen test samples
# these predictions will be compared against the real `yTest` to measure performance

print("The mean squared error = %.2f" % mean_squared_error(yTest, yPredTest))
# calculates the Mean Squared Error (MSE) between predictions and true test values
# Smaller MSE -> better model fit
# if MSE = 0 -> predict predictions (rare)

'''
Summary


'''