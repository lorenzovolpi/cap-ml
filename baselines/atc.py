import numpy as np 
from sklearn.metrics import f1_score

def get_entropy(probs): 
	return np.sum( np.multiply(probs, np.log(probs + 1e-20))  , axis=1)

def get_max_conf(probs):
	return np.max(probs, axis=-1)
	
def find_ATC_threshold(scores, labels): 
    sorted_idx = np.argsort(scores)
    
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    fp = np.sum(labels==0)
    fn = 0.0
    
    min_fp_fn = np.abs(fp - fn)
    thres = 0.0
    for i in range(len(labels)): 
        if sorted_labels[i] == 0: 
            fp -= 1
        else: 
            fn += 1
        
        if np.abs(fp - fn) < min_fp_fn: 
            min_fp_fn = np.abs(fp - fn)
            thres = sorted_scores[i]
    
    return min_fp_fn, thres


def get_ATC_acc(thres, scores): 
    return np.mean(scores>=thres)

def get_ATC_f1(thres, scores, probs):
    preds = np.argmax(probs, axis=-1)
    estim_y = abs(1 - (scores>=thres)^preds)
    return f1_score(estim_y, preds)
    