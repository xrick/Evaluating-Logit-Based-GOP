from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

def calculate_metrics(y_true, gop_scores, threshold=0):
    """
    Calculates classification metrics based on GOP scores.
    As per paper, a higher GOP score indicates a better pronunciation.
    A negative score suggests a mispronunciation.
    So, we predict 'mispronounced' (0) if GOP < threshold, and 'correct' (1) otherwise.
    
    Args:
        y_true (list): The ground truth labels (1 for correct, 0 for mispronounced).
        gop_scores (list): The calculated GOP scores.
        threshold (float): The threshold to classify scores.
    
    Returns:
        A dictionary of metrics.
    """
    y_pred = [1 if score >= threshold else 0 for score in gop_scores]
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    return metrics