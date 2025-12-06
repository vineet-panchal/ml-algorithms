import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
ccdefault = pd.read_csv(r'ccdefault.csv')

# Separate the class from the data attributes
y = ccdefault['DEFAULT']
X = ccdefault.drop(['DEFAULT', 'ID'], axis = 1)

#  Split the data into training and test sets
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

# Normalize the data
scaler = Normalizer()
xTrain_norm = scaler.fit_transform(xTrain)

# Conduct kNN classification
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(xTrain_norm, yTrain)

# Apply attribute transform to the test instances
xTest_norm = scaler.transform(xTest)
# Predict the labels of the test instances
yPred = clf.predict(xTest_norm)

# Report the accuracy of the classifier on the test data
print("Accuracy:", round(accuracy_score(yTest, yPred)*100), '%')