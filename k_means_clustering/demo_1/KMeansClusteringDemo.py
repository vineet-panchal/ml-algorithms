
'''
Dataset: UCI Wholesale Customers: https://archive.ics.uci.edu/dataset/292/wholesale+customers

Background
The Wholesale Customers dataset refers to clients of a wholesale distributor. It includes the annual
spending data from clients in monetary units on diverse product categories. I obtained the dataset from
UC Irvine’s Machine Learning Repository and I looking for datasets with tasks for classification and
clustering. The dataset includes 440 instances and 8 attributes for spending on Channel, Region, Fresh,
Milk, Grocery, Frozen, Detergents_Paper, and Delicassen [1]. The dataset is interesting because it
represents real-world customer segmentation, different buying patterns, and market analysis. Identifying
groups within this dataset could help businesses understand their customers better and allows for better
decision making.

Methods
I started off by importing necessary libraries including pandas, matplotlib.pyplot, and StandardScalar,
KMeans, AgglomerativeClustering, silhouette_score, PCA from sklearn. I then downloaded the dataset,
and read it using pandas dataframe. The data required preprocessing by normalizing the numeric values
because some of the features had very large values while others were very small. I used standard scalar to
scale the dataset. I then performed two clustering algorithms: K-Means Clustering and Hierarchical
Agglomerative Clustering. I used K-Means clustering to group customers into k segments based on their
spending. I used Agglomerative clustering to compare results between two different clustering algorithms.

Results
I used the silhouette score to select the appropriate number of clusters. I used PCA to reduce 6D high
dimensional data to 2D to visualize how clusters separate. I then plotted two scatter plots, one two
visualize the two clustering methods, and the other two visualize the silhouette scores. K-Means showed
optimal performance around k=3 with a silhouette score around 0.45. K-Means clustering showed
different purchasing patterns. The PCA visualization showed separated clusters, confirming validity of
segmentation.

Conclusion
The wholesale customers dataset represents natural segmentation that can support business decisions.
K-Means with k=3 produced the best results. Preprocessing processes like scaling improved clustering.
This analysis demonstrates how clustering can uncover meaningful patterns for instance customer groups.
'''


'''
The Goal: to use clustering to visualize natural customer segments

Steps: 
1. Load the dataset -> read the wholesale customers dataset
2. Preprocessing -> standardize the data so that all numeric values have a mean of 0, and standard deviation of 1
  - some of the features have very large values while others are very small, so we have to make them all comparable in scale
3. K-Means Clustering -> group customers into k segments based on spending, find patterns for high grocery spending, low spenders, etc.
4. Agglomerative Clustering -> compare results between two different clustering algorithms
5. PCA Visualization -> to reduce the 6D data to 2D to visualize how clusters separate in a scatter plot.
6. Plot figures -> use matplotlib to plot the two figures, one for K-Means clustering, and the other for Silhouette scores
7. Silhouette Score -> tells us which k is the best
'''

import pandas as pd # to load the data
import matplotlib.pyplot as plt # to plot the graphs
from sklearn.preprocessing import StandardScaler # to standarize inputs
from sklearn.cluster import KMeans, AgglomerativeClustering # algorithms for clustering
from sklearn.metrics import silhouette_score # to evaluate the clustering
from sklearn.decomposition import PCA # to reduce dimensions for visualization

# 1. Load the dataset
df = pd.read_csv("wholesale_customers_data.csv") # reading the dataset into a Pandas dataframe

# 2. Preprocessing
scaler = StandardScaler() # creates a scaler that standardizes each feature to mean=0 and variance=1
X_scaled = scaler.fit_transform(df.iloc[:, 2:])
# fits the scalar to the data and applies it
# selects columns starting from index 2 because columns 0 and 1 are Region and Channel

# 3. K-Means clustering
sil_scores = [] # list to store the scores
K_range = range(2, 10) # test from k = 2 to k = 9
for k in K_range: # loop over different k values
    kmeans = KMeans(n_clusters=k, random_state=42) # creates a KMeans object
    labels = kmeans.fit_predict(X_scaled) # fit the model and get the cluster labels
    sil_scores.append(silhouette_score(X_scaled, labels)) # measure how well the clusters are sperated and add it to the list

# fitting the best model
kmeans = KMeans(n_clusters=3, random_state=42) # based on the scores, k=3 is typically a good choice for the dataset
labels_k = kmeans.fit_predict(X_scaled) # fit and get the labels

# 4. Hierarchical clustering
agg = AgglomerativeClustering(n_clusters=3, linkage="ward") # minimizes variance within clusters
labels_h = agg.fit_predict(X_scaled) # assigns each point a cluster label

# 5. PCA visualization
pca = PCA(n_components=2) # create the transformer
X_pca = pca.fit_transform(X_scaled) # compress the data

# 6. Plot the figures
plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_k)
plt.title("K-Means Clusters (PCA)")
plt.savefig("k_means_clustering.png")
plt.show()

plt.figure()
plt.plot(list(K_range), sil_scores)
plt.title("Silhouette Scores")
plt.savefig("silhouette_scores.png")
plt.show()

# 7. Silhouette Score
print("The Silhouette score for k=3:", silhouette_score(X_scaled, labels_k))