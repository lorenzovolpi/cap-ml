import quapy as qp

def from_name(err_name):
    if err_name == 'f1e':
        return f1e
    else:
        return qp.error.from_name(err_name)
    
def f1e(prev):
    return 1 - f1_score(prev)

def f1_score(prev):
    recall = prev[0] / (prev[0] + prev[1])
    precision = prev[0] / (prev[0] + prev[2])
    return 2 * (precision * recall) / (precision + recall)
