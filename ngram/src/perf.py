from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import pandas as pd
import sys
import numpy as np




def printMetric(precision, recall, specificity):
        print('PRECISION '+str(precision))
        print('RECALL '+str(recall))
        print('Specificity  '+str(specificity))



def metricsAtTS(y_score, y_true, TS):
    eps=0.00000001
    tp = len(y_true[y_score >= TS][y_true[y_score >= TS]==1])
    fp = len(y_true[y_score >= TS][y_true[y_score >= TS]==0])
    tn = len(y_true[y_score < TS][y_true[y_score < TS]==0])
    fn = len(y_true[y_score < TS][y_true[y_score < TS]==1])
    return float(tp) / float(tp + fp + eps), float(tp) / float(tp + fn + eps), float(tn) / float(tn + fp + eps)


def scoresForMemo(y_score, y_true):
    eps=0.00000001
    tp = len(y_true[y_score == 1][y_true[y_score == 1] == 1])
    fp = len(y_true[y_score == 1][y_true[y_score == 1] == 0])
    tn = len(y_true[y_score == 0][y_true[y_score == 0] == 0])
    fn = len(y_true[y_score == 0][y_true[y_score == 0] == 1])
    precision_memo, recall_memo, specificiy_memo = float(tp) / float(tp + fp + eps), \
                                                   float(tp) / float(tp + fn + eps), \
                                                   float(tn) / float(tn + fp + eps)
    return precision_memo, recall_memo, specificiy_memo

def computeBestTS(y_score, y_true, precision_memo, specificiy_memo):
    for TS in np.arange(0.7, 1.0, 0.001):
        precision, recall, specificity = metricsAtTS(y_score,
                                                    y_true, TS)
        #print('Threshold ' + str(TS))
        #printMetric(precision, recall, specificity)

        if precision>precision_memo and specificity>specificiy_memo:
            return precision, recall, specificity, TS
    return precision, recall, specificity, TS


filename = sys.argv[1]
predictions = pd.read_csv(filename,sep='\t')
predictions.columns=[['keyword','name','descr','brand','cat','in_memo','annotation','prediction']]
#predictions.columns=[['keyword','name','descr','brand','skuid','annotation','prediction']]

#Score memo
y_score  = predictions.in_memo.as_matrix()
y_true = predictions.annotation.as_matrix()
#print (y_true)
#y_true[y_true == '1'] = 1
#y_true[y_true == '0'] = 0


y_true[y_true > 0] = 1
y_true[y_true <= 0] = 0


precision_memo, recall_memo, specificiy_memo =  scoresForMemo(y_score, y_true)
print('Score for Memo')
printMetric(precision_memo, recall_memo, specificiy_memo)

y_score  = predictions.prediction.as_matrix()
print('Score for Letters')
precision, recall, specificity, TS = computeBestTS(y_score, y_true, precision_memo, specificiy_memo)

print('Threshold '+str(TS))
printMetric(precision, recall, specificity)




