import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(preds, targets):

    best_thresh = 0.5

    best_f1 = 0

    for t in np.arange(0.3, 0.71, 0.05):

        binary = (preds >= t).astype(np.uint8)

        score = f1_score(
            targets.flatten(),
            binary.flatten()
        )

        if score > best_f1:
            best_f1 = score
            best_thresh = t

    return best_thresh, best_f1