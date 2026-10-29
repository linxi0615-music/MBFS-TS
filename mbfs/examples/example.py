"""Minimal usage example for the MBFS library."""

import numpy as np
from sklearn.datasets import make_classification

from mbfs import MBFSThompson, evaluate_selection

X, y = make_classification(
    n_samples=400,
    n_features=200,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    random_state=0,
)

model = MBFSThompson(
    n_features=X.shape[1],
    n_episodes=100,
    alpha_td=0.9,
    gamma=0.9,
    max_episode_steps=X.shape[1],
    early_stopping_patience=30,
    reward_estimator="logistic_l1",
    estimator_C=0.05,
    task="classification",
)

results = model.fit_predict(X, y, verbose=True)

metrics = evaluate_selection(results["theta"], true_idx=list(range(5)))
print(metrics)