
# Demo Preprocessing (partial)
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
data = pd.read_table(r'data.txt')

# Check the size of the data
shapeBefore = data.shape

# Check for all unique values for a given attribute
data.country.unique()
# Drop instances that have any missing data
data.dropna(inplace=True)
data = data[data.country != '(nu']

# PCA analysis
pca5 = PCA(n_components=5)
X = data.iloc[:,7:].values
reducedData = pca5.fit_transform(X)

# Consolidate the data
reducedData = pd.DataFrame(
  reducedData, 
  columns= ['PC1','PC2','PC3','PC4','PC5']
)

reducedData = pd.concat([
  data.iloc[:,0:7].reset_index(drop=True),
  reducedData.reset_index(drop=True)
], axis = 1)

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(reducedData['PC1'], reducedData['PC2'], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: PC1 vs PC2')
plt.grid(True)
plt.savefig("preprocessing-demo1.png")
plt.show()