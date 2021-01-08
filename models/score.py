"""
Score an experiment
"""

from sklearn.metrics import roc_curve
import matplotlib as plt
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

def plot_roc(y_true: List[str], y_score: List[float]) -> None:
    """Plot a ROC curve for these results"""

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label="target")
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print("EER: {}%".format(EER*100))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



if __name__=='__main__':

    import sys

    t, s = read_result_file(sys.argv[1])
    score = eer(t, s)

    print("EER: {}%".format(score*100))




