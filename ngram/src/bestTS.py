from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import pandas as pd
import sys
import numpy as np


def printMetric(tp, fp, tn, fn, doprint):
    eps=0.00000001
    if doprint:
        print('TP '+str(tp))
        print('FP '+str(fp))
        print('TN '+str(tn))
        print('FN '+str(fn))
        print('PRECISION '+str(float(tp)/float(tp+fp+eps)))
        print('RECALL '+str(float(tp)/float(tp+fn+eps)))
        print('Specificity  '+str(float(tn)/float(tn+fp+eps)))
    return float(tp)/float(tp+fp+eps), float(tp)/float(tp+fn+eps), float(tn)/float(tn+fp+eps)


#
# return precision, recall, and specificity
#
def printForTS(y_score, y_true, TS, doprint):
    tp = len(y_true[y_score >= TS][y_true[y_score >= TS]==1])
    fp = len(y_true[y_score >= TS][y_true[y_score >= TS]==0])
    tn = len(y_true[y_score < TS][y_true[y_score < TS]==0])
    fn = len(y_true[y_score < TS][y_true[y_score < TS]==1])
    return printMetric(tp, fp, tn, fn, doprint)


def computeBestTS(y_score, y_true, precision_memo, recall_memo, specificiy_memo):
    for TS in np.arange(0.0, 1.0, 0.001):
        precision, recall, specificity = printForTS(y_score,
                                                    y_true, TS, False)
        if precision>precision_memo and specificity>specificiy_memo:
            printForTS(y_score,y_true, TS, True)
            return TS

def scoresForMemo(y_score, y_true):
    tp = len(y_true[y_score == 1][y_true[y_score == 1] == 1])
    fp = len(y_true[y_score == 1][y_true[y_score == 1] == 0])
    tn = len(y_true[y_score == 0][y_true[y_score == 0] == 0])
    fn = len(y_true[y_score == 0][y_true[y_score == 0] == 1])
    precision_memo, recall_memo, specificiy_memo = printMetric(tp, fp, tn, fn, False)
    return precision_memo, recall_memo, specificiy_memo



filename = sys.argv[1]
print(filename)
predictions = pd.read_csv(filename,sep='\t')
predictions.columns=[['keyword','name','descr','brand','cat','in_memo','annotation','prediction']]
y_score = predictions.in_memo.as_matrix()
y_true = predictions.annotation.as_matrix()
y_true[y_true > 0] = 1
y_true[y_true <= 0] = 0

precision_memo, recall_memo, specificiy_memo = scoresForMemo(y_score, y_true)
y_score = predictions.prediction.as_matrix()
TS = computeBestTS(y_score, y_true, precision_memo, recall_memo, specificiy_memo)
print(TS)









