import numpy as np

def precision_recall_at_k(actual, predicted, k=10):
    pred_k = predicted[:k]
    actual_set = set(actual)
    pred_set = set(pred_k)
    intersect = len(actual_set & pred_set)
    precision = intersect / k
    recall = intersect / len(actual_set) if actual_set else 0
    return precision, recall
