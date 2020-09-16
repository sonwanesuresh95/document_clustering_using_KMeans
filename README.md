# document-clustering-using-KMeans
observing behaviour of <b>KMeans clustering algorithm</b> on a real-world text documents dataset.

## Info
This is a project for <b>observing</b> how a clustering algorithm like <b>kmeans</b> works with <b>text documents</b>.<br><br>
<b>Dataset</b> used for this project is <b>20_newsgroups dataset</b>.<br> <br>
This 20_newsgroups dataset comprises around <b>18000 newsgroups posts on 20 topics</b> which can be used for training <b>supervised as well as unsupervised learning algorithms</b> for machine learning tasks like <b>text classification, document classification, document clustering, topic modelling</b> etc. <br><br>
<b>Reference</b> : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html<br><br>

The script <b>train_kmeans.py</b> contains code that runs <b>KMeans</b> clustering algorithm consecutively <b>from 2 to 20 clusters</b>.<br>
It <b>produces benchmarks</b> of <b>training time</b> taken by the algorithm, <b>Silhouettes scores</b> and <b>inertias</b> on each step of configuration.<br>
Prints out the results to CLI, <b>generates benchmark plots</b> and saves them in <b>./images</b><br><br>

## Usage
For installing requirements, do<br>
<code>
$pip install requirements.txt
</code><br><br>
For downloading 20_newsgroup dataset, do<br>
<code>
$python download_dataset.py<br>
</code><br><br>
For producing clustering benchmarks on 20_newsgroup dataset, do<br>
<code>
$python train_kmeans.py<br>
</code><br><br>

## Review : Plots of benchmarks
### Plot of n_clusters vs Training Time
![](https://github.com/sonwanesuresh95/document-clustering-using-KMeans/blob/master/images/training%20time.png "training time")
### Plot of n_clusters vs Silhouettes score
![](https://github.com/sonwanesuresh95/document-clustering-using-KMeans/blob/master/images/silhouettes%20score.png "silhouettes score")
### Plot of n_clusters vs Inertia
![](https://github.com/sonwanesuresh95/document-clustering-using-KMeans/blob/master/images/inertia.png "inertias")

