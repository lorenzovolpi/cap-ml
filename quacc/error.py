import quapy as qp

def from_name(err_name):
    if err_name == 'f1e':
        return f1e
    elif err_name == 'f1':
        return f1
    else:
        return qp.error.from_name(err_name)
    
# def f1(prev):
#     # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
#     if prev[0] == 0 and prev[1] == 0 and prev[2] == 0:
#         return 1.0
#     elif prev[0] == 0 and prev[1] > 0 and prev[2] == 0:
#         return 0.0
#     elif prev[0] == 0 and prev[1] == 0 and prev[2] > 0:
#         return float('NaN')
#     else:
#         recall = prev[0] / (prev[0] + prev[1])
#         precision = prev[0] / (prev[0] + prev[2]) 
#         return 2 * (precision * recall) / (precision + recall)

def f1(prev):
    den = (2*prev[3]) + prev[1] + prev[2]
    if den == 0:
        return 0.0
    else:
        return (2*prev[3])/den

def f1e(prev):
    return 1 - f1(prev)

def acc(prev):
    return (prev[1] + prev[2]) / sum(prev)