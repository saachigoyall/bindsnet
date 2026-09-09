"""Cross-validated decoder accuracy for whatever a run_*.py script produced.

Uses 5-fold stratified CV rather than a single train/test split: with n=150 a
30% holdout leaves ~45 test samples, and the confidence interval on that is wide
enough (~+-15 points) to hide the effect we are looking for.

    python analyze.py layer_decoder.json
    python analyze.py weight_sweep.json
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def accuracy(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return cross_val_score(clf, np.array(X), np.array(y),
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()


data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'layer_decoder.json'))
first = next(iter(data.values()))

if 'Xin' in first:                                   # layer_decoder.json
    print(f"{'decode pole direction from...':<36}" +
          ''.join(f'seed {s:<7}' for s in data) + 'mean')
    for key, label in [('Xin', 'INPUT layer spike counts'),
                       ('Xhid', 'HIDDEN layer spike counts')]:
        a = [accuracy(d[key], d['y']) for d in data.values()]
        print(f'{label:<36}' + ''.join(f'{v:<12.3f}' for v in a) + f'{np.mean(a):.3f}')
else:                                                # weight_sweep.json
    print(f"{'w_ih max':<12}" + ''.join(f'seed {s:<7}' for s in first) + 'mean')
    for w, per_seed in data.items():
        a = [accuracy(d['Xhid'], d['y']) for d in per_seed.values()]
        print(f'{w:<12}' + ''.join(f'{v:<12.3f}' for v in a) + f'{np.mean(a):.3f}')

print('\nchance = 0.500')
