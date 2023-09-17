import numpy as np

def get_doc(probs1, probs2):
    return np.mean(probs2) - np.mean(probs1) 