"""Synthetic ground-truth generators for the G1-G8 sensitivity gates.

This is the ONLY module in the analysis path that draws random numbers; gate.py
and report.py are kept free of RNG so their outputs can never carry a random
placeholder (enforced by a CI grep). Every generator takes an integer ``seed``
and is fully reproducible.

Model: P = sigmoid(a*(theta - b)), M ~ Bernoulli(P).
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _responses(rng, a, b, theta):
    p = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    return (rng.random((theta.size, a.size)) < p).astype(float)


def scenario_separated(seed, thetas, n_items=600, a=1.5):
    rng = make_rng(seed)
    theta = np.asarray(thetas, dtype=float)
    avec = np.full(n_items, float(a))
    b = rng.normal(0.0, 1.0, n_items)
    return {"scores": _responses(rng, avec, b, theta)}


def scenario_null(seed, n_models=6, n_items=300, a=1.0):
    rng = make_rng(seed)
    theta = np.zeros(n_models)
    avec = np.full(n_items, float(a))
    b = rng.normal(0.0, 1.0, n_items)
    return {"scores": _responses(rng, avec, b, theta)}


def scenario_two_equal(seed, n_items=400, a=1.5):
    rng = make_rng(seed)
    theta = np.zeros(2)
    avec = np.full(n_items, float(a))
    b = rng.normal(0.0, 1.0, n_items)
    return {"scores": _responses(rng, avec, b, theta)}


def scenario_dead(seed, n_models=120, n_items=60, k=6):
    rng = make_rng(seed)
    a = rng.uniform(0.8, 2.0, n_items)
    b = rng.normal(0.0, 1.0, n_items)
    theta = rng.normal(0.0, 1.0, n_models)
    scores = _responses(rng, a, b, theta)
    dead_cols = rng.choice(n_items, size=k, replace=False)
    for c in dead_cols:
        scores[:, c] = (rng.random(n_models) < 0.5).astype(float)
    return {"scores": scores, "dead_cols": sorted(int(c) for c in dead_cols)}


def scenario_mislabel(seed, n_models=120, n_items=60, k=4):
    rng = make_rng(seed)
    a = rng.uniform(0.8, 2.0, n_items)
    b = rng.normal(0.0, 1.0, n_items)
    theta = rng.normal(0.0, 1.0, n_models)
    scores = _responses(rng, a, b, theta)
    order = np.argsort(theta)
    cols = rng.choice(n_items, size=k, replace=False)
    half = n_models // 2
    for c in cols:
        scores[order[half:], c] = 0.0  # able models "fail"
        scores[order[:half], c] = 1.0  # weak models "pass"
    return {"scores": scores, "mis_cols": sorted(int(c) for c in cols)}


def scenario_saturation(seed, n_models=120, n_items=60):
    rng = make_rng(seed)
    a = np.full(n_items, 1.5)
    b = rng.uniform(-3.0, -1.0, n_items)  # all easy
    theta = rng.uniform(-2.0, 3.0, n_models)
    return {"scores": _responses(rng, a, b, theta)}


def coverage_truth(theta_a=0.4, theta_b=0.0, a=1.5, n=200000, seed=99999):
    rng = make_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    pa = 1.0 / (1.0 + np.exp(-a * (theta_a - b)))
    pb = 1.0 / (1.0 + np.exp(-a * (theta_b - b)))
    return float(np.mean(pa - pb))


def coverage_sample(seed, theta_a=0.4, theta_b=0.0, a=1.5, n_items=300):
    rng = make_rng(seed)
    b = rng.normal(0.0, 1.0, n_items)
    avec = np.full(n_items, float(a))
    theta = np.array([theta_a, theta_b])
    return {"scores": _responses(rng, avec, b, theta)}
