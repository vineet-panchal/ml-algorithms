### PART 1 ###

# Step 1: Importing required libraries
from sklearn import datasets # gives access to built-in datasets (like breast cancer wisconsin dataset)
import matplotlib.pyplot as plt # used for plotting the decision tree and graphs
from sklearn import tree # contains DecisionTreeClassifier and plot_tree
from sklearn.metrics import accuracy_score # evaluates model acccuracy
from sklearn.model_selection import train_test_split, GridSearchCV # split data and tune model parameters

# Step 2: Load the breast cancer wisconsin dataset
X, y = datasets.load_breast_cancer(return_X_y = True) #(4 points) 
# X contains the features, and y contains the target class labels (0 for malignant, 1 for benign)

# Step 3: Check Dataset Size
print("There are", X.shape[0], "instances described by", X.shape[1], "features.")
# Check how many instances we have in the dataset, and how many features describe these instances

# Step 4: Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify = y, random_state = 42)
# test_size=0.4 ensures 40% of data is in the test set
# stratify = y ensures that both training and test sets have approximately the same percentage of instances of each target class as the complete set
# random_state = 42 ensures reproducibility of the split
# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# In addition, ensure that the training and test sets # contain approximately the same 
# percentage of instances of each target class as the complete set.

# Step 5: Create the Decision Tree Classifier
clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split = 6)
clf.fit(X_train, y_train) 
# criterion = "entropy" means we use entropy to measure the quality of a split
# min_samples_split = 6 means that nodes with less than 6 training instances are not further split
# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# To measure the quality of a split, using the entropy criteria.
# Ensure that nodes with less than 6 training instances are not further split

# Step 6: Predict and Evaluate Accuracy
predC = clf.predict(X_test) 
# Apply the decision tree to classify the data 'testData'.
print('The accuracy of the classifier is', accuracy_score(y_test, predC))
# Compute the accuracy of the classifier on 'testData'
# this compares predicted vs. actual labels and prints accuracy (usually around 93-96%)

# Step 7: Visualize the Decision Tree
plt.figure(figsize = (20,10))
_ = tree.plot_tree(clf, filled = True, fontsize = 12)  
plt.show()
# draws the deicsion tree (big diagram showing nodes, splits, and class labels)


### PART 2.1 ###
# Step 1: Create Lists for Accuracy
trainAccuracy = []
testAccuracy = [] 
depthOptions = range(1, 16) 
# Visualize the training and test accuracies as a function of the maximum depth of the decision tree
# Initialize 2 empty lists where you will save the training and testing accuracies 
# as we iterate through the different decision tree depth options.
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees

# Step 2: Loop Over Depth Options
for depth in depthOptions:
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
    # Also, ensure that nodes with less than 6 training instances are not further split
    cltree = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split = 6, max_depth = depth) 
    # Decision tree training
    cltree.fit(X_train, y_train) 
    # Label predictions on training set 
    y_predTrain = cltree.predict(X_train) 
    # Label predictions on test set 
    y_predTest = cltree.predict(X_test)
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) 
# trains multiple trees (with different max depths)
# records accuracy for both training and test sets for each depth

# Step 3: Plot Accuracy vs Depth
plt.plot(depthOptions, trainAccuracy, marker = 'o', color = 'blue', label = 'Training Accuracy') #(3 points) 
plt.plot(depthOptions, testAccuracy, marker = 's', color = 'red', label = 'Test Accuracy') #(3 points)
plt.legend(['Training Accuracy','Test Accuracy']) # add a legend for the training accuracy and test accuracy (1 point) 
plt.xlabel('Tree Depth') # name the horizontal axis 'Tree Depth' (1 point) 
plt.ylabel('Classifier Accuracy') # name the vertical axis 'Classifier Accuracy' (1 point) 
plt.show()
# Plot of training and test accuracies vs the tree depths (use different markers of different colors)
# you'll see that training accuracy increases with depth, but test accuracy may peak and then decrease (overfitting)

""" 
According to the test accuracy, the best model to select is when the maximum depth is equal to 5 or 6, approximately. 
But, we should not use select the hyperparameters of our model using the test data, because it leads to information leakage, test data should be used for final evaluation, not for tuning.
"""

### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth and the minimum number of samples to split a node
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# Step 1: Define the Grid
parameters = {'max_depth': range(1, 16), 'min_samples_split': range(2, 11)}
# Define the parameters to be optimized: the max depth of the tree and the minimum number of samples to split a node
# try all combinations of depths (1-15) and splits (2-10)

# Step 2: Grid Search
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion = "entropy")) 
clf.fit(X_train, y_train) 
# We will still grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
# runs cross-validation for each parameter combination
# picks the one with the highest validation accuracy

# Step 3: Display Best Parameters
tree_model = clf.best_estimator_
print("The maximum depth of the tree sis", clf.best_params_['max_depth'], 
      'and the minimum number of samples required to split a node is', clf.best_params_['min_samples_split'])
# The best model is tree_model. Visualize that decision tree (tree_model). Set the font size the 12 

# Step 4: Visualize the Best Decision Tree
plt.figure(figsize = (20,10))
_ = tree.plot_tree(tree_model, filled = True, fontsize = 12) 
plt.show()
# draws the tuned model's final decision tree.

""" 
This method for tuning the hyperparameters of our model is acceptable, because it uses cross-validation. 
"""

# Explain below what is tenfold Stratified cross-validation?
"""
Tenfold Stratified cross-validation is a technique used to evaluate the performance of a machine learning model. 
It involves dividing the dataset into ten equal parts or "folds" while ensuring that each fold maintains the same class distribution as the entire dataset.
______
"""

