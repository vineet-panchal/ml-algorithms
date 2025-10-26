'''
This demo is using Wine dataset to show how a Decision Tree Classifier works in scikit-learn.

Essentially, this demo: 

1. Loads the dataset
2. Splits it into training and testing parts
3. Trains a decision tree to classify wine types
4. Tests the model's accuracy

This demo is a real implementation of the 2 - Supervised Learning: Classification Notes

The Wine Dataset:
- this dataset contains measurements of different chemical properties of wine (like alcohol, magnesium, phenols, etc.)
- there is a label that says which type of wine it is (Class 1, 2, or 3)
'''


import pandas as pd # for loading and handling tabular data (like Excel sheets)
from sklearn import tree # the module in scikit-learn that provides decision tree algorithms
from sklearn.model_selection import train_test_split # function to randomly split data into training and testing sets
from sklearn.metrics import accuracy_score # function to calculate how accurate the model's predictions are
import matplotlib.pyplot as plt # matplotlib to visualize our decision tree

data = pd.read_csv(r"wine.csv", header = None)
# this reads the wine dataset from a .csv file
# `pd.read_csv()` -> loads a CSV file into pandas DataFrame
# `header=None` -> tells pandas that the CSV file doesn't have a header row (column names)
# so, column = 0 is the first column in the dataset, column = 1, is the second

train, test = train_test_split(data, test_size = 0.3)
# splits the dataset into training and testing sets:
# - ***train*** -> 70% of the data (used to train the model)
# - ***test*** -> 30% (used to evaluate accuracy)

yTrain = train.iloc[:,0]
yTest = test.iloc[:,0]
# extract the target variable (the class label)
# `.loc[:, 0]` -> all rows, but only the first column (index 0)
# in the wine dataset, the first column is the Wine Class (1, 2, or 3)
# `yTrain` = target labels for training
# `yTest` = target labels for testing

xTrain = train.iloc[:,1:]
xTest = test.iloc[:,1:]
# extracts all the data (input) attributes
# `.iloc[:, 1:]` -> all columns except the first (from index 1 onward)
# these columns are numerical values like: alcohol, malic acid, ash, magnesium, flavonoids, etc.
# `xTrain` -> training data (wine attributes)
# `xTest` -> testing data (wine attributes)

clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
# this creates the decision tree classifier (the model)
# `criterion="entropy"` -> the tree will use information gain to decide how to split nodes
# you can specify which methods to split nodes, for example using the gini index: criterion="gini"
# `max_depth=3` -> limits how deep the tree can grow to avoid overfitting
# to summarize this line of code says:
# "Create a decision tree classifier that splits based on entropy and has a maximum of 3 levels."

clf.fit(xTrain, yTrain)
# this trains the model/classifier using the training data
# * The algorithm looks for patterns between the attributes (`xTrain`) and labels (`yTrain`)
# * it builds a decision tree by splitting nodes where entropy decreases the most (highest information gain)

yPred = clf.predict(xTest)
# applies the trained model (the decision tree) to the test data (to classify the test data)
# `yPred` -> stores the predictions

feature_names = [f"feature_{i}" for i in range(1, xTrain.shape[1] + 1)]
class_names = [str(c) for c in sorted(yTrain.unique())]
fig = plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
plt.title("Decision Tree")
plt.show()
# display the decision tree

print("The accuracy of the classifier is", accuracy_score(yTest, yPred))
# compute the accuracy of the classifer on the test data, it evaluates how good the model is.
# `accuracy_score()` -> compare the predicted labels (`yPred`) with the true labels (`yTest`)
# it returns a number between 0 and 1 (percentage) showing how accurate the predictions were

'''
Summary:

1. Import libraries -> load pandas, scikit-learn, matplotlib, etc.
2. Load dataset -> read 'wine.csv' into DataFrame
3. Split data -> 70% train, 30% test
4. Extract features/labels/attributes -> seperate attributes (x) and class labels (y)
5. Create Decision Tree -> use entropy and limit depth to 3
6. Train model -> fit on training data
7. Predict test data -> classify unseen wines
8. Evaluate model -> calculate accuracy score


Connection to Supervised Learning: Classification: 

Classification Task -> predict wine class (1, 2, or 3)
Training vs Test sets -> `train_test_split()`
Attributes & Labels -> `xTrain`, `yTrain`, `xTest`, `yTest`
Entropy -> `criterion="entropy"
Information Gain -> used internally by sklearn's tree algorithm, determines each split
Hunt's Algorithm -> implmented recursively inside scikit-learn's tree builder
Stopping Criteria (max depth) -> `max_depth=3`
Overfitting -> prevented by limiting tree depth
Classification Output -> `class = ...` at each leaf node
Model Evaluation -> `accuracy_score()`


How The Model Decides:

At each node:
1. It checks which attribute split gives the largest information gain (reduces entropy most).
2. It uses that attribute to divide the data into two branches
3. This repeats until:
  - the tree reaches max_depth = 3, or
  - the node is pure (entropy = 0)

So, the root node is the most important feature for classification, it explains the biggest difference between wine types.
'''