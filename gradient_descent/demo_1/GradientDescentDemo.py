'''
This demo is meant to show Gradient Descent, how we can manually compute and update regression parameters without using scikit-learn

Essentially, this demo:

1. Creates a small dataset (X, y)
2. Starts with inital guesses for regression parameters, β0, β1
3. Uses Gradient Descent to update them step by step
4. Plots the line of best fit at the end

This demo is the real implementation of Gradient Descent from 3 - Supervised Learning: Model Overfitting, Regression, and Gradient Descent.
'''

import numpy as np # for arrays and math operations (like vector multiplication)
import matplotlib.pyplot as plt # for plotting data and the regression line

X = np.array([55,60,65,70,75,80])
y = np.array([316,292,268,246,227,207])
# creates two 1D numpy arrays representing data points

beta0 = 0
beta1 = 0
# building the model, these are the model parameters
# `beta0` -> intercept (y-intercept)
# `beta1` -> slope of the line
# both are initialized to 0, we will update them using gradient descent

a = 0.00001 #learning rate alpha
n_iterations = 1000000
# `a` -> learning rate, controls how big each update step is.
# too large -> unstable (overshoots the minimum)
# too small -> very slow convergence
# `n_iterations` -> number of times we'll update `beta0` and `beta1`
# so, the code will loop 1 million times to minimize the cost

for i in range(n_iterations):
# perform gradient descent
# this is where core gradient descent logic happens
  
    yPred = beta0 + beta1*X
    # step 1: compute predictions
    # predicts current y-values given current parameters
    
    dBeta0 = -2*sum(y-yPred) # Gradient of cost function wrt beta0
    dBeta1 = -2*sum(X*(y-yPred)) #Gradient of cost function wrt beta1
    # step 2: compute gradients
    # we need the derivative of the cost function with respect to each parameter
    # the two lines above measure the slope (gradient) of the error surface for each parameter
    
    beta0 = beta0 - a*dBeta0
    beta1 = beta1 - a*dBeta1
    # gradient descent update step
    # we move in the opposite direction of the gradient to reduce the cost
    # each iteration slightly adjusts β0 and β1 toward better values
    
    print("Iteration", i, ', beta0 =', beta0, ', beta1 =', beta1)
    # print the parameter values after each iteration so we can watch them converge

yPred = beta0 + beta1*X
# after training, recalculate final predicted line (`yPred`)

plt.scatter(X,y)
plt.plot(X, yPred, color = 'r')
# plot the results
# black points is the actual data
# red line is the regression line found via gradient descent
# we should see a downward-sloping line (since y decreases as X increases)

plt.show()
# display it