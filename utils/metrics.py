import numpy as np


def compute_accuracy(tp, tn, fn, fp):  # only fit 2 cls condition
    '''
    Accuracy = TP + TN / FP + FN + TP + TN
    '''
    return ((tp + tn) * 100) / float(tp + tn + fn + fp)


def compute_specificity(tn, fp):
    '''
    Precision = TN  / FN + TP
    '''
    return (tn * 100) / float(tn + fp)


def compute_sensitivity(tp, fn):  # equal to recall
    '''
    Recall = TP /FN + TP
    '''
    return (tp * 100) / float(tp + fn)


def compute_precision(tp, fp):  # equal to Positive Predictive Value(PPV)
    '''
    Precision = TP  / FP + TP
    '''
    return (tp * 100) / float(tp + fp)


def compute_recall(tp, fn):  # equal to Sensitivity
    '''
    Recall = TP /FN + TP
    '''
    return (tp * 100) / float(tp + fn)


def compute_f1_score(tp, tn, fp, fn):
    # calculates the F1 score
    precision = compute_precision(tp, fp)/100
    recall = compute_recall(tp, fn)/100
    f1_score = (2*precision*recall)/ (precision + recall)
    return f1_score * 100


def compute_NPV(tn, fn):  # Negative Predictive Value
    '''
    Negative Predictive Value = tn  / tn + fn
    '''
    return (tn * 100) / float(tn + fn)







