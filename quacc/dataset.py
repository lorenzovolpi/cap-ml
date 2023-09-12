import quapy as qp

def get_imdb_traintest():
    return qp.datasets.fetch_reviews("imdb", tfidf=True).train_test

def get_spambase_traintest():
    return qp.datasets.fetch_UCIDataset("spambase", verbose=False).train_test