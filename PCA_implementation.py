import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.datasets import load_iris


# Bogging your head to properly import the iris data-set 
iris = datasets.load_iris()
df = pd.DataFrame(iris.data , columns = iris.feature_names)
df["target"] = iris.target

# Writing and Implementing PCA on the iris dataset 
X = np.array(df.iloc[: , :-1]) 
X_mean = np.mean(X , axis = 0) 
X_centr = X - X_mean

B = np.dot(np.transpose(X_centr) , X_centr) / (X_centr.shape[0]-1)
# print(B) # B is the covariance matrix 

# Now we need to do the eigen value decomposition of this covariance matrix 
eigvalues , eigvectors = np.linalg.eig(B)

# Sort the eigen values and corresponding eigenvectors 
sorted_indices = np.argsort(eigvalues)[::-1]
eigvalues = eigvalues[sorted_indices]
eigvectors = eigvectors[:, sorted_indices]

'''
Since each column in eigenvectors corresponds to an eigenvalue in eigenvalues, 
you need to reorder the columns in the same order as the sorted eigenvalues.
eigenvectors[:, sorted_] reorders the columns of eigenvectors based on the sorted indices.

'''
print("Eigenvalues:", eigvalues)
print("Eigenvectors:\n", eigvectors)

# Now we have the directions from the eigen values and vectors, so we have to project the data 
X_projected = np.dot(X_centr , eigvectors[: , :2])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=df["target"], cmap="viridis", edgecolor="k", s=50)
plt.title("PCA from Scratch")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


                                        ## FOR 3D PLOTTING -- taken from gpt ## 
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

# Project data onto the first 3 components for scratch PCA
X_pca_scratch_3d = np.dot(X_centr, eigvectors[:, :3])

# Plotting both results in 3D for comparison
fig = plt.figure(figsize=(14, 6))

# Plot scratch PCA in 3D
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(X_pca_scratch_3d[:, 0], X_pca_scratch_3d[:, 1], X_pca_scratch_3d[:, 2],
                  c=df["target"], cmap="viridis", edgecolor="k", s=50)
ax1.set_title("PCA from Scratch (3D)")
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.set_zlabel("Principal Component 3")
plt.show()