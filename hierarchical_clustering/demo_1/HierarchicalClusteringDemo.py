
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'vertebrate.csv')

# Drop the attributes 'name'and 'class'
X = data.drop(['Name', 'Class'], axis = 1)
Z = hierarchy.average(X)

# Plot the dendrogram
dn = hierarchy.dendrogram(Z, labels = data['Name'].to_list())
plt.show()

k = 5
label = hierarchy.fcluster(Z, k, criterion = 'maxclust')

# Concatenate the dataframe 'data' with the array of cluster labels
data = pd.concat((data, pd.DataFrame(label, columns = ['Labels'])), axis = 1)