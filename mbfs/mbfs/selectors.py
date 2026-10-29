"""MBFS: MDP + Thompson Sampling feature selection."""

from __future__ import annotations

import time
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

__all__ = ["MBFSThompson"]


def _build_classifier(
    kind: str, C: float, random_state: int, n_classes: int
):
    """Build a sklearn classifier for reward estimation."""
    if kind == "logistic_l1":
        kwargs = dict(
            penalty="l1",
            solver="liblinear",
            C=C,
            random_state=random_state,
            max_iter=1000,
        )
        if n_classes > 2:
            try:
                return LogisticRegression(multi_class="ovr", **kwargs)
            except TypeError:
                # sklearn >= 1.5 removed multi_class parameter
                return LogisticRegression(**kwargs)
        return LogisticRegression(**kwargs)
    if kind == "linear_svc_l1":
        return LinearSVC(
            penalty="l1",
            dual=False,
            C=C,
            random_state=random_state,
            max_iter=1000,
        )
    if kind == "svm_linear":
        return SVC(
            kernel="linear",
            C=C,
            random_state=random_state,
            max_iter=1000,
        )
    if kind == "svm_rbf":
        return SVC(
            kernel="rbf",
            gamma="scale",
            C=C,
            random_state=random_state,
        )
    raise ValueError(f"Unknown reward_estimator: {kind}")


class MBFSThompson:
    """MDP + Thompson Sampling feature selector.

    Parameters
    ----------
    n_features : int
        Total number of features ``p``.
    n_episodes : int, default 200
        Number of training episodes.
    alpha_td : float, default 0.1
        TD learning rate.
    gamma : float, default 0.9
        Discount factor.
    beta_prior : tuple of float, default (1.0, 1.0)
        Beta prior ``(a0, b0)`` for each feature.
    random_state : int, default 42
        Random seed.
    max_episode_steps : int, default 50
        Maximum features added per episode.
    early_stopping_patience : int, default 10
        Stop after this many episodes without selected-set change.
    reward_estimator : {'logistic_l1', 'linear_svc_l1',
                        'svm_linear', 'svm_rbf'}, default 'logistic_l1'
        Classifier used to compute the reward for classification tasks.
    estimator_C : float, default 0.1
        Regularization inverse strength of the reward estimator.
    task : {'classification', 'regression'}, default 'classification'
        Task type.

    Attributes
    ----------
    alpha_ : np.ndarray of shape (n_features,)
        Posterior Beta ``alpha`` after fitting.
    beta_ : np.ndarray of shape (n_features,)
        Posterior Beta ``beta`` after fitting.
    theta_ : np.ndarray of shape (n_features,)
        Posterior mean ``alpha / (alpha + beta)`` after fitting.
    """

    def __init__(
        self,
        n_features: int,
        n_episodes: int = 200,
        alpha_td: float = 0.1,
        gamma: float = 0.9,
        beta_prior: Tuple[float, float] = (1.0, 1.0),
        random_state: int = 42,
        max_episode_steps: int = 50,
        early_stopping_patience: int = 10,
        reward_estimator: str = "logistic_l1",
        estimator_C: float = 0.1,
        task: str = "classification",
    ):
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )

        self.n_features = n_features
        self.n_episodes = n_episodes
        self.alpha_td = alpha_td
        self.gamma = gamma
        self.beta_prior = beta_prior
        self.random_state = random_state
        self.max_episode_steps = max_episode_steps
        self.early_stopping_patience = early_stopping_patience
        self.reward_estimator_type = reward_estimator
        self.estimator_C = estimator_C
        self.task = task

        np.random.seed(random_state)

        # Beta posterior parameters per feature.
        self.alpha = np.full(n_features, beta_prior[0])
        self.beta = np.full(n_features, beta_prior[1])

        # TD state-value cache and performance cache.
        self.state_value_cache: dict = {}
        self.performance_cache: dict = {}

        # Internal data.
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._scaler: Optional[StandardScaler] = None

        # Fitted attributes.
        self.alpha_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Reward helpers
    # ------------------------------------------------------------------ #
    def compute_adjusted_r2(self, selected_idx) -> float:
        """Adjusted R^2 of the selected feature subset (regression)."""
        key = tuple(sorted(selected_idx))
        if key in self.performance_cache:
            return self.performance_cache[key]

        if len(selected_idx) == 0:
            self.performance_cache[key] = 0.0
            return 0.0

        X_sel = self._X.iloc[:, selected_idx]
        regressor = LinearRegression()

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )
        r2_scores = []

        for train_idx, val_idx in kf.split(X_sel):
            X_train, X_val = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
            y_train, y_val = self._y.iloc[train_idx], self._y.iloc[val_idx]

            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_val)

            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            n = len(y_val)
            p = len(selected_idx)
            if n > p + 1:
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            else:
                adjusted_r2 = r2

            r2_scores.append(adjusted_r2)

        mean_adjusted_r2 = float(np.mean(r2_scores))
        self.performance_cache[key] = mean_adjusted_r2
        return mean_adjusted_r2

    def compute_accuracy(self, selected_idx) -> float:
        """Cross-validated accuracy of the selected feature subset."""
        key = tuple(sorted(selected_idx))
        if key in self.performance_cache:
            return self.performance_cache[key]

        if len(selected_idx) == 0:
            unique, counts = np.unique(self._y, return_counts=True)
            majority = unique[np.argmax(counts)]
            accuracy = float(np.mean(self._y == majority))
            self.performance_cache[key] = accuracy
            return accuracy

        X_sel = self._X.iloc[:, selected_idx]
        n_classes = int(len(np.unique(self._y)))

        classifier = _build_classifier(
            kind=self.reward_estimator_type,
            C=self.estimator_C,
            random_state=self.random_state,
            n_classes=n_classes,
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_val_score(
            classifier, X_sel, self._y, cv=cv, scoring="accuracy"
        )
        accuracy = float(np.mean(scores))

        self.performance_cache[key] = accuracy
        return accuracy

    def compute_performance(self, selected_idx) -> float:
        """Return the reward signal for a feature subset."""
        if self.task == "regression":
            return self.compute_adjusted_r2(selected_idx)
        return self.compute_accuracy(selected_idx)

    # ------------------------------------------------------------------ #
    # State value helpers
    # ------------------------------------------------------------------ #
    def get_state_value(self, desc) -> float:
        """Return V(s); 0 for unseen states."""
        key = tuple(sorted(desc))
        return self.state_value_cache.get(key, 0.0)

    def set_state_value(self, desc, value: float) -> None:
        """Assign V(s)."""
        key = tuple(sorted(desc))
        self.state_value_cache[key] = value

    # ------------------------------------------------------------------ #
    # One episode
    # ------------------------------------------------------------------ #
    def run_episode(self) -> None:
        """Run a single episode: forward feature chain with TD updates."""
        max_steps = min(self.max_episode_steps, self.n_features)

        current_desc: list = []
        available = list(range(self.n_features))
        np.random.shuffle(available)

        no_improvement_count = 0
        max_no_improvement = 5

        for _ in range(max_steps):
            if len(available) == 0:
                break

            # Thompson sampling.
            samples = np.random.beta(
                self.alpha[available], self.beta[available]
            )
            best_local = int(np.argmax(samples))
            f = available[best_local]

            s_desc = current_desc
            s_prime_desc = sorted(current_desc + [f])

            current_v = self.get_state_value(s_desc)
            next_v = self.get_state_value(s_prime_desc)

            current_perf = self.compute_performance(s_desc)
            next_perf = self.compute_performance(s_prime_desc)
            reward = next_perf - current_perf

            # TD update.
            td_target = reward + self.gamma * next_v
            updated_current_v = current_v + self.alpha_td * (
                td_target - current_v
            )
            self.set_state_value(s_desc, updated_current_v)

            # Posterior update.
            if td_target > updated_current_v:
                self.alpha[f] += 1
            else:
                self.beta[f] += 1

            current_desc = s_prime_desc
            available.pop(best_local)

            if reward <= 0:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    break
            else:
                no_improvement_count = 0

        # Anchor terminal state with its true performance.
        if (
            len(current_desc) > 0
            and self.get_state_value(current_desc) == 0.0
        ):
            self.set_state_value(
                current_desc, self.compute_performance(current_desc)
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X,
        y,
        verbose: bool = True,
    ) -> "MBFSThompson":
        """Fit the selector on data ``(X, y)``."""
        start_time = time.time()

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._X = pd.DataFrame(
            X_scaled, columns=list(range(self.n_features))
        )
        self._y = y if isinstance(y, pd.Series) else pd.Series(y)

        self.state_value_cache = {}
        self.performance_cache = {}
        self.alpha = np.full(self.n_features, self.beta_prior[0])
        self.beta = np.full(self.n_features, self.beta_prior[1])

        best_features = set()
        epochs_without_improvement = 0
        best_theta = None

        iterator = tqdm(
            range(self.n_episodes),
            desc="MBFS+TS",
            disable=not verbose,
        )

        for ep in iterator:
            self.run_episode()

            theta = self.alpha / (self.alpha + self.beta)
            selected = set(np.where(theta > 0.5)[0])

            if len(selected) == 0 and self.n_features > 0:
                k = min(5, self.n_features)
                selected = set(np.argsort(theta)[-k:])

            if selected == best_features:
                epochs_without_improvement += 1
            else:
                best_features = selected
                best_theta = theta.copy()
                epochs_without_improvement = 0

            if verbose and (ep + 1) % 20 == 0:
                iterator.set_postfix(
                    {
                        "θ>0.5": len(selected),
                        "max_θ": f"{np.max(theta):.3f}",
                        "NoImpr": epochs_without_improvement,
                    }
                )

            if epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(
                        f"\nEarly stop: no change for "
                        f"{self.early_stopping_patience} episodes"
                    )
                break

        if best_theta is not None:
            self.theta_ = best_theta
        else:
            self.theta_ = self.alpha / (self.alpha + self.beta)

        self.alpha_ = self.alpha.copy()
        self.beta_ = self.beta.copy()
        self.elapsed_ = time.time() - start_time

        if verbose:
            print(f"\nTraining done | time: {self.elapsed_:.2f}s")
            print(f"mean theta: {np.mean(self.theta_):.4f}")

        return self

    def fit_predict(self, X, y, verbose: bool = True) -> dict:
        """Fit and return ``theta`` / ``alpha`` / ``beta`` / ``time``."""
        self.fit(X, y, verbose=verbose)
        return {
            "theta": self.theta_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "time": self.elapsed_,
        }

    def get_selected_features(self, threshold: float = 0.5) -> list:
        """Return indices with posterior mean > ``threshold``."""
        if self.theta_ is None:
            raise RuntimeError("Call fit() before get_selected_features().")
        return np.where(self.theta_ > threshold)[0].tolist()