'''
You are a recruiter and your goal is to predict whether an applicant is likely to get hired or rejected.
You have gathered data over the years that you intend to use as a training set. Your task is to use logistic
regression to build a model that predicts whether an applicant is likely to be hired or not, based on the
results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.
'''


import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt


# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
data = pandas.read_csv("LogisticRegressionData.csv", header = None, names=['Score1', 'Score2', 'y'])

# Seperate the data features (score1 and Score2) from the class attribute 
X = data[['Score1', 'Score2']]
y = data['y']

# Plot the data using a scatter plot to visualize the data. 
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]])
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression()
regS.fit(X, y)

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X)
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] #this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[y_pred[i]], color = c[y_pred[i]])
fig.canvas.draw()
plt.show()
# Notice that some of the training instances are not correctly classified. These are the training errors.