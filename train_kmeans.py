import time

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing import get_tfidf
from download_dataset import path
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

# set random state = 0
np.random.seed(0)

# categories to perform KMeans
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# load dataset
dataset = fetch_20newsgroups(data_home=path, categories=categories)
print('Fetched dataset successfully')
features = dataset.data
target = dataset.target
print('number of text documents = {}\n'.format(len(features)))
del dataset

# Get features of dataset using Tfidf
tfidf = TfidfVectorizer()
print('Generating Tfidf features from text data')
features = tfidf.fit_transform(features)
print('Tfidf features generated successfully')
print('Shape of Tfidf sparse feature matrix = {}\n'.format(features.shape))

# Specify params for clustering
n_clusters = 20

# benchmark lists
silhouettes = []
inertias = []
times = []

# Run kmeans 20 times consecutively on n_clusters ranging from 2 to 20
for i in range(n_clusters + 1)[2:]:
    print('\nRunning Kmeans clutering with n_clusters = {}'.format(i))
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=1, max_iter=100)
    start_time = time.time()
    kmeans.fit(features)
    time_taken = time.time() - start_time
    print('Training done. Took {} seconds'.format(time_taken))
    times.append(time_taken)
    silhouettes.append(silhouette_score(features, kmeans.labels_))
    inertias.append(kmeans.inertia_)
    print('Benchmarks on KMeans with n_clusters = {}'.format(i))
    print('Training Time     = {} seconds'.format(time_taken))
    print('Silhouettes Score = {}'.format(silhouette_score(features, kmeans.labels_)))
    print('Inertia           = {}'.format(kmeans.inertia_))

print('Done Training, Now producing benchmarks..\n')

# Plot results of Clustering
# Plot n_clusters vs Training time
print('Generating plot n_clusters vs Training time')
plt.figure()
plt.plot(range(2, 21), times, marker='o', label='Training Time')
plt.grid(True)
plt.xticks(range(2, 21), range(2, 21))
plt.xlabel('n_clusters')
plt.ylabel('Training times (seconds)')
plt.gca().set_ylim([0, 2.5])
plt.gca().set_xlim([2, 20])
plt.title('n_clusters vs Training times\n')
plt.legend(loc='lower right')
plt.savefig('./images/training time.png')
print('Saved plot in ./images as training time.png\n')

# plot n_clusters vs silhouettes score
print('Generating plot n_clusters vs silhouettes score')
plt.figure()
plt.plot(range(2, 21), silhouettes, marker='o', label='Silhouette Score')
plt.xticks(range(2, 21), range(2, 21))
plt.gca().set_xlim([2, 20])
plt.grid(True)
plt.xlabel('n_clusters')
plt.ylabel('Silhouettes score')
plt.title('n_clusters vs Silhouette score\n')
plt.gca().set_xlim([2, 20])
plt.legend(loc='lower right')
plt.savefig('./images/silhouettes score.png')
print('Saved plot in ./images as silhouettes score.png\n')

# Plot n_clusters vs Inertia
print('Generating plot n_clusters vs Inertia')
plt.figure()
plt.plot(range(2, 21), inertias, marker='o', label='inertia')
plt.xticks(range(2, 21), range(2, 21))
plt.xlabel('n_clusters')
plt.ylabel('Inertia')
plt.title('n_clusters vs Inertia\n')
plt.gca().set_xlim([2, 20])
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('./images/inertia.png')
print('Saved plot in ./images as inertia.png')