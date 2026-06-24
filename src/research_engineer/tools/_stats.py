"""Pure-Python statistics helpers for Phase 8.

Implements mean, variance, Welch's t-test, two-tailed p-value via the
regularized incomplete beta function (numerically stable continued
fraction), Cohen's d, and 95% confidence intervals. No SciPy dependency.
"""

from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    """Arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: list[float], ddof: int = 1) -> float:
    """Sample variance (ddof=1) or population (ddof=0)."""
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / (n - ddof)


def std(values: list[float], ddof: int = 1) -> float:
    """Standard deviation."""
    return math.sqrt(variance(values, ddof))


def welch_t_statistic(a: list[float], b: list[float]) -> float:
    """Welch's t-statistic for two independent samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    va, vb = variance(a), variance(b)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0:
        return 0.0
    return (ma - mb) / denom


def welch_degrees_of_freedom(a: list[float], b: list[float]) -> float:
    """Welch-Satterthwaite degrees of freedom."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = variance(a), variance(b)
    term_a = (va / na) ** 2 / (na - 1)
    term_b = (vb / nb) ** 2 / (nb - 1)
    denom = term_a + term_b
    if denom == 0:
        return float(na + nb - 2)
    return (va / na + vb / nb) ** 2 / denom


def _ln_beta(a: float, b: float) -> float:
    """Log of the Beta function via lgamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, max_iter: int = 200,
            eps: float = 1e-14) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b).

    Returns value in [0, 1]. Uses the continued-fraction expansion from
    Numerical Recipes. Symmetry I_x(a,b) = 1 - I_{1-x}(b,a) is used
    when x > (a+1)/(a+b+2) for numerical stability.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _ln_beta(a, b)
    front = math.exp(
        math.log(x) * a + math.log(1.0 - x) * b - lbeta
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution with df degrees of freedom."""
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    ib = regularized_incomplete_beta(df / 2.0, 0.5, x)
    if t > 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_p_value_two_tailed(t: float, df: float) -> float:
    """Two-tailed p-value for Student's t."""
    if df <= 0:
        return 1.0
    cdf = student_t_cdf(t, df)
    p = 2.0 * min(cdf, 1.0 - cdf)
    return max(0.0, min(1.0, p))


def inverse_student_t_cdf(p: float, df: float,
                          tol: float = 1e-8,
                          max_iter: int = 200) -> float:
    """Inverse CDF (quantile) of Student's t via bisection on the CDF.

    p is the cumulative probability (0..1).
    """
    if df <= 0:
        return 0.0
    if p <= 0.0:
        return -1e6
    if p >= 1.0:
        return 1e6
    lo, hi = -100.0, 100.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        c = student_t_cdf(mid, df)
        if abs(c - p) < tol:
            return mid
        if c < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def t_critical_two_tailed(df: float, alpha: float = 0.05) -> float:
    """Two-tailed critical t value for a given alpha (e.g. 0.05 -> 95% CI)."""
    return inverse_student_t_cdf(1.0 - alpha / 2.0, df)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d with pooled standard deviation."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    va, vb = variance(a), variance(b)
    pooled_var = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return (ma - mb) / pooled_std


def effect_size_label(d: float) -> str:
    """Label for Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def mean_diff_ci_95(
    a: list[float], b: list[float], df: float
) -> list[float]:
    """95% confidence interval for the mean difference (mean_a - mean_b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return [0.0, 0.0]
    ma, mb = mean(a), mean(b)
    va, vb = variance(a), variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return [ma - mb, ma - mb]
    tcrit = t_critical_two_tailed(df, alpha=0.05)
    diff = ma - mb
    return [diff - tcrit * se, diff + tcrit * se]


def welch_p_value(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (t_statistic, two_tailed_p_value) for Welch's t-test."""
    t = welch_t_statistic(a, b)
    df = welch_degrees_of_freedom(a, b)
    return t, student_t_p_value_two_tailed(t, df)
