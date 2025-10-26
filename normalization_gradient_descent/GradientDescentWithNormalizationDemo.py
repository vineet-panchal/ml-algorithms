'''
This demo builds directly on top of the Gradient Descent demo, but it adds one very important concept: feature scaling (or normalization).

Essentially, this demo:

1. Uses the same dataset (X, y) as before (simple 1D data)
2. applies standardization (z-score normalization) to scale both X and y
3. Performs gradient descent on the scaled data (which converges faster)
4. transforms predictions back to the original scale for visualization

This demo shows the importance of feature scaling for gradient-based optimization.
This demo is a real implementation of Gradient Descent with Normalization from 3 - Supervised Learning: Model Overfitting, Regression, and Gradient Descent.
'''

import numpy as np # math + arrays
import matplotlib.pyplot as plt # visualization
from sklearn.preprocessing import StandardScaler # from scikit-learn, used for z-score normalization

X = np.array([55,60,65,70,75,80]).reshape(-1,1)
y = np.array([316,292,268,246,227,207]).reshape(-1,1)
# same dataset as the gradient descent demo
# `.reshape(-1, 1)` -> makes them 2D column vectors, required by `StandardScaler`.
# with 2D vectors we can later compute mean, std, and perform matrix-like operations

scalerX = StandardScaler()
scalerX.fit(X)
scalerY = StandardScaler()
scalerY.fit(y)
# 1. Create `StandardScaler` objects for X and y
# 2. `.fit()` calculates: 
# - mean (μ)
# - standard deviation (σ)

xScaled = scalerX.transform(X)
yScaled = scalerY.transform(y)
# transforms each feature using `x(scaled) = x - μ / σ`
# this means, mean becomes 0, and standard deviation becomes 1, Why?:
# Gradient descent converges much faster when features are scaled, because the cost function contours become circular instead of stretched

beta0 = 0
beta1 = 0
# building the model, start β₀ (intercept) and β₁ (slope) at 0.

a = 0.001 #learning rate alpha
n_iterations = 1000
# `a` -> learning rate = 0.001 (larger than before because data is scaled).  
# Run 1000 iterations of gradient descent.

for i in range(n_iterations):
# perform gradient descent

    yPred = beta0 + beta1*xScaled
    # step 1: compute predictions
    
    dBeta0 = -2*sum(yScaled-yPred) # Gradient of cost function wrt beta0
    dBeta1 = -2*sum(xScaled*(yScaled-yPred)) #Gradient of cost function wrt beta1
    # step 2: compute partial derivatives
    # these are the gradients that tell the direction to move to minimize error
    
    beta0 = beta0 - a*dBeta0
    beta1 = beta1 - a*dBeta1
    # gradient descent update
    # So each iteration moves β₀ and β₁ *opposite to the gradient direction* to reduce cost.
    
    print("Iteration", i, ', beta0 =', beta0, ', beta1 =', beta1)
    # to see how they converge
    # with scaling, you'll notice convergence happens much faster than the gradient descent demo

yPred = beta0 + beta1*xScaled
# after training, compute the final predictions for the scaled data
# at this point, `yPred` is still in standardized units, not the original y-scale

plt.scatter(X,y)
plt.plot(X, scalerY.inverse_transform(yPred), color = 'r')
# plot the results, original data as black points
# plot the regression line in red

plt.show()    
# display it