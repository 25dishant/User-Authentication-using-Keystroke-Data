from sklearn.metrics import roc_curve
import numpy as np


def evaluateEER(user_scores, imposter_scores):

     labels = [0]*len(user_scores) + [1]*len(imposter_scores)
     fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores,pos_label=1)
#    print 'fpr','tpr','thres',fpr,tpr,thresholds
     missrates = 1 - tpr
     farates = fpr
#    array = np.zeros((123,3))
#    array[:,0] = missrates
#    array[:,1] = farates
#    array[:,2] = thresholds
#    print array
     dists = missrates - farates
     idx1 = np.argmin(dists[dists >= 0])
     idx2 = np.argmax(dists[dists < 0])
     x = [missrates[idx1], farates[idx1]]
     y = [missrates[idx2], farates[idx2]]
     if (( y[1] - x[1] - y[0] + x[0] ) == 0):
            a = 0
     else:
            a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
     eer = x[0] + a * ( y[0] - x[0] )
     return eer
