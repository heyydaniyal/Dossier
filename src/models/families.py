"""P4 model families behind one interface. Label-free module: y arrives as an argument.

Every model exposes, for a matrix X whose columns are `features` (registry names, in order):
  margin(X)         raw score; logistic and LightGBM: log-odds margin of the UNCALIBRATED model
  proba(X)          sigmoid(margin) for logistic / LightGBM; None for the rule scores
  contributions(X)  (per-feature contribution matrix, bias): exact and additive, rows sum to margin
  to_dict()         everything needed to rebuild it (json-serialisable); from_dict() rebuilds it
Explanations are in the margin space (log-odds), never in calibrated-probability units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fired_cols(features: list[str]) -> list[int]:
    idx = [i for i, f in enumerate(features) if f.startswith("fired_")]
    if not idx:
        raise ValueError("rule scores need the fired_* features")
    return idx


# ---------------------------------------------------------------- rules


@dataclass
class RulesCount:
    """Score = number of rules fired (no fitting)."""

    features: list[str]
    family: str = "rules_count"
    params: dict = field(default_factory=dict)

    def fit(self, X, y, eval_set=None):
        _fired_cols(self.features)
        return self

    def contributions(self, X):
        c = np.zeros_like(np.asarray(X, float))
        j = _fired_cols(self.features)
        c[:, j] = np.asarray(X, float)[:, j]
        return c, 0.0

    def margin(self, X):
        c, b = self.contributions(X)
        return c.sum(axis=1) + b

    def proba(self, X):
        return None

    def to_dict(self):
        return {"family": self.family, "features": self.features, "params": self.params}

    @classmethod
    def from_dict(cls, d):
        return cls(features=d["features"], params=d["params"])


@dataclass
class RulesWeighted:
    """Score = sum over fired rules of log(lift), lift = smoothed precision / prevalence, both on
    the fit rows."""

    features: list[str]
    family: str = "rules_weighted"
    params: dict = field(default_factory=lambda: {"smoothing": 1.0})
    weights: dict = field(default_factory=dict)

    def fit(self, X, y, eval_set=None):
        X, y = np.asarray(X, float), np.asarray(y)
        a = float(self.params["smoothing"])
        prev = (y.sum() + a) / (len(y) + 2 * a)
        self.weights = {}
        for j in _fired_cols(self.features):
            on = X[:, j] == 1
            prec = (y[on].sum() + a) / (on.sum() + 2 * a)
            self.weights[self.features[j]] = float(math.log(prec / prev))
        return self

    def contributions(self, X):
        X = np.asarray(X, float)
        c = np.zeros_like(X)
        for j, f in enumerate(self.features):
            if f in self.weights:
                c[:, j] = X[:, j] * self.weights[f]
        return c, 0.0

    def margin(self, X):
        c, b = self.contributions(X)
        return c.sum(axis=1) + b

    def proba(self, X):
        return None

    def to_dict(self):
        return {
            "family": self.family,
            "features": self.features,
            "params": self.params,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(features=d["features"], params=d["params"], weights=d["weights"])


# ---------------------------------------------------------------- logistic


def signed_log1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


@dataclass
class Logistic:
    """L2 logistic regression. x -> sign(x) log1p|x|, standardised with fit-row mean/sd; NaN -> 0
    after standardising (= the fit mean) plus a missing flag for every column with NaN in the fit
    rows. A flag's coefficient is folded into its feature's contribution."""

    features: list[str]
    params: dict
    family: str = "logistic"
    state: dict = field(default_factory=dict)

    def _design(self, X):
        X = signed_log1p(np.asarray(X, float))
        mu, sd = np.array(self.state["mean"]), np.array(self.state["scale"])
        Z = (X - mu) / sd
        miss = np.isnan(Z)
        Z = np.where(miss, 0.0, Z)
        flags = miss[:, self.state["flag_cols"]].astype(float)
        return Z, flags

    def fit(self, X, y, eval_set=None):
        Xt = signed_log1p(np.asarray(X, float))
        mu = np.nanmean(Xt, axis=0)
        sd = np.nanstd(Xt, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where(~np.isfinite(sd) | (sd == 0), 1.0, sd)
        flag_cols = [int(j) for j in np.flatnonzero(np.isnan(Xt).any(axis=0))]
        self.state = {"mean": mu.tolist(), "scale": sd.tolist(), "flag_cols": flag_cols}
        Z, F = self._design(X)
        cw = None if self.params["class_weight"] == "none" else "balanced"
        m = LogisticRegression(
            C=self.params["C"], class_weight=cw, max_iter=self.params["max_iter"]
        )
        m.fit(np.hstack([Z, F]), np.asarray(y))
        p = Z.shape[1]
        self.state |= {
            "coef": m.coef_[0][:p].tolist(),
            "coef_flag": m.coef_[0][p:].tolist(),
            "intercept": float(m.intercept_[0]),
            "n_iter": int(m.n_iter_[0]),
        }
        return self

    def contributions(self, X):
        Z, F = self._design(X)
        c = Z * np.array(self.state["coef"])
        for k, j in enumerate(self.state["flag_cols"]):
            c[:, j] += F[:, k] * self.state["coef_flag"][k]
        return c, self.state["intercept"]

    def margin(self, X):
        c, b = self.contributions(X)
        return c.sum(axis=1) + b

    def proba(self, X):
        return sigmoid(self.margin(X))

    def to_dict(self):
        return {
            "family": self.family,
            "features": self.features,
            "params": self.params,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(features=d["features"], params=d["params"], state=d["state"])


# ---------------------------------------------------------------- LightGBM


@dataclass
class LightGBM:
    """Gradient-boosted trees. With eval_set: early stopping on that (time-later) set, metric
    average precision only. Without: exactly params['n_estimators'] trees."""

    features: list[str]
    params: dict  # LightGBM params incl. fixed protocol params; class_weight 'none' | 'balanced'
    family: str = "lightgbm"
    early_stopping_rounds: int = 100
    model_str: str = ""
    best_iteration: int = 0
    _booster: object = field(default=None, repr=False)

    def _lgb_params(self) -> dict:
        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        p["class_weight"] = None if cw == "none" else "balanced"
        p["metric"] = "average_precision"
        return p

    def fit(self, X, y, eval_set=None):
        m = lgb.LGBMClassifier(**self._lgb_params())
        if eval_set is not None:
            Xe, ye = eval_set
            m.fit(
                np.asarray(X, float),
                np.asarray(y),
                eval_X=(np.asarray(Xe, float),),
                eval_y=(np.asarray(ye),),
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            self.best_iteration = int(m.best_iteration_)
        else:
            m.fit(np.asarray(X, float), np.asarray(y))
            self.best_iteration = int(m.booster_.current_iteration())
        self.model_str = m.booster_.model_to_string(num_iteration=self.best_iteration)
        self._booster = lgb.Booster(model_str=self.model_str)
        return self

    def _b(self):
        if self._booster is None:
            self._booster = lgb.Booster(model_str=self.model_str)
        return self._booster

    def margin(self, X):
        return self._b().predict(np.asarray(X, float), raw_score=True)

    def proba(self, X):
        return sigmoid(self.margin(X))

    def contributions(self, X):
        c = self._b().predict(np.asarray(X, float), pred_contrib=True)
        return c[:, :-1], c[:, -1]

    def to_dict(self):
        return {
            "family": self.family,
            "features": self.features,
            "params": self.params,
            "best_iteration": self.best_iteration,
            "model_str": self.model_str,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            features=d["features"],
            params=d["params"],
            model_str=d["model_str"],
            best_iteration=d["best_iteration"],
        )


FAMILIES = {
    c.__dataclass_fields__["family"].default: c
    for c in (RulesCount, RulesWeighted, Logistic, LightGBM)
}


def from_dict(d: dict):
    return FAMILIES[d["family"]].from_dict(d)
