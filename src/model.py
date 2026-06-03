"""Train/test split, logistic regression model, backtest, leakage check."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle as sk_shuffle


SPLIT_DATE = "2025-01-01"
LAGS_FOR_MODEL = [1]  # predict next-day SPX direction from same-day crypto returns


def make_features(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Features: BTC[t] and ETH[t].
    Label: sign of SPX[t+1] (1=up, 0=down/flat).
    No future data leaks into features: SPX[t+1] is obtained by shifting SPX back by 1.
    """
    X = returns[["BTC-USD", "ETH-USD"]].copy()
    y = (returns["^GSPC"].shift(-1) > 0).astype(int)
    # Drop last row: SPX[t+1] is NaN for the final date
    combined = pd.concat([X, y.rename("label")], axis=1).dropna()
    X_clean = combined[["BTC-USD", "ETH-USD"]]
    y_clean = combined["label"]
    return X_clean, y_clean


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    split_date: str = SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split on date — no shuffling."""
    mask_train = X.index < split_date
    mask_test = X.index >= split_date
    return X[mask_train], y[mask_train], X[mask_test], y[mask_test]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Fit logistic regression on training data only."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def backtest(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    spx_test_returns: pd.Series,
) -> dict:
    """Report directional accuracy, F1, and comparison to naive baseline."""
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    naive_accuracy = float((y_test == 1).mean())  # always predict "up"
    coef = dict(zip(X_test.columns, model.coef_[0]))
    results = {
        "accuracy": accuracy,
        "f1": f1,
        "naive_accuracy": naive_accuracy,
        "n_test": len(y_test),
        "coefficients": coef,
        "y_pred": y_pred,
        "y_test": y_test,
        "spx_test_returns": spx_test_returns.loc[y_test.index],
    }
    print("\n=== Backtest Results ===")
    print(f"  Test period: {y_test.index[0].date()} → {y_test.index[-1].date()}  (n={len(y_test)})")
    print(f"  Directional accuracy: {accuracy:.1%}")
    print(f"  Naive baseline (always up): {naive_accuracy:.1%}")
    print(f"  F1 score: {f1:.3f}")
    print(f"  Coefficients: {coef}")
    return results


def check_for_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_shuffles: int = 100,
) -> float:
    """
    Permutation test: shuffle training labels, retrain, test on real labels.

    Expected result: shuffled model achieves ≈ naive accuracy (majority class rate),
    confirming the real model's signal (if any) comes from features, not leakage.
    With class imbalance, the expected shuffled accuracy is the majority class rate,
    not 50%.
    """
    naive_acc = float((y_test == 1).mean())
    shuffled_accs = []
    for _ in range(n_shuffles):
        y_shuffled = sk_shuffle(y_train.values, random_state=None)
        m = LogisticRegression(max_iter=1000, random_state=42)
        m.fit(X_train, y_shuffled)
        shuffled_accs.append(accuracy_score(y_test, m.predict(X_test)))
    mean_acc = float(np.mean(shuffled_accs))
    print(f"\n=== Leakage Check (permutation test) ===")
    print(f"  Naive baseline (majority class): {naive_acc:.1%}")
    print(f"  Shuffled-training-label accuracy (mean over {n_shuffles} runs): {mean_acc:.1%}")
    print(f"  Expected: close to naive baseline {naive_acc:.1%}")
    passed = abs(mean_acc - naive_acc) < 0.05
    print(f"  Leakage check: {'PASSED' if passed else 'FAILED — investigate!'}")
    return mean_acc
