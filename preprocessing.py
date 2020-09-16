import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf(features):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(features)
