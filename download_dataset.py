import sklearn
from sklearn.datasets import fetch_20newsgroups

path = './data'


# Download and cache dataset
def download_dataset():
    dataset = fetch_20newsgroups(data_home=path)
    print('Dataset 20_newsgroups downloaded successfully')


if __name__ == '__main__':
    download_dataset()
