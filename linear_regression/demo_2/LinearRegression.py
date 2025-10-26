'''
Description:
- You are working for a restaurant franchise. The franchise wants to open new restaurants in various cities.

Goal:
- Predict the profit of a restaurant based on the population of the city where it is located.
- The data is provided in the file RegressionData.csv.
- The chain already has several restaurants in different cities.
- Model the relationship between profit and population using linear regression.

Steps: 
- Load the data from RegressionData.csv.
- Visualize the data using a scatter plot.
- Train a linear regression model using the training data.
- Use the trained model to predict the profit of a restaurant in a city with a population of 18.
'''

import os
print("Current working directory:", os.getcwd())
os.chdir("/Users/vineetpanchal/Desktop/MY_STUFF/VPrograms/ml-algorithms/linear_regression/demo_2")

import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 


# Open the csv file RegressionData.csv in Excel, notepad++ or any other applications to have a 
# rough overview of the data at hand. 
# You will notice that there are several instances (rows), of 2 features (columns). 
# The values to be predicted are reported in the 2nd column.

# Load the data from the file RegressionData.csv in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'X' and the second feature 'y' (these are the labels)
data = pandas.read_csv("RegressionData.csv", header = None, names=['X', 'y'])
# Reshape the data so that it can be processed properly
X = data['X'].values.reshape(-1,1) 
y = data['y'] 
# Plot the data using a scatter plot to visualize the data
plt.scatter(X, y) 

# Linear regression using least squares optimization
reg = linear_model.LinearRegression()
reg.fit(X, y) 

# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) 
plt.scatter(X, y, c='b') 
plt.plot(X, y_pred, 'r') 
fig.canvas.draw()
plt.show()

print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_, " and the weight b_1 is equal to ", reg.coef_[0])

# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
print("the profit/loss in a city with 18 habitants is ", reg.predict([[18]]))