import quapy as qp

def getImdbTrainTest():
    return qp.datasets.fetch_reviews("imdb", tfidf=True).train_test