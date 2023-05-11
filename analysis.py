from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
dataset = pd.read_csv('features_data_1.csv')
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
print(scaled_data)
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
print(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.show()
