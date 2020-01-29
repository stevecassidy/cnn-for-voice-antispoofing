"""
Score an experiment
"""

from sklearn.metrics import roc_curve
import numpy as np

from typing import List, Tuple


def read_result_file(filename: str) -> Tuple[List[str], List[float]]: 
    """Read result lines, return two lists of true labels and scores"""

    true = []
    scores = []
    with open(filename) as fd:
        for line in fd:
            score, target = line.split()
            true.append(target)
            scores.append(float(score))

    return true, scores

def eer(y_true: List[str], y_score: List[float]) -> float: 
    """Compute the Equal Error Rate for a result list"""

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label="target")
    fnr = 1 - tpr

    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return EER


if __name__=='__main__':

    import sys

    t, s = read_result_file(sys.argv[1])
    score = eer(t, s)

    print("EER: {}%".format(score*100))




