import numpy as np

def snv(X):
    mu = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - mu) / std

def msc(X, ref=None):
    H, W, C = X.shape
    if ref is None:
        ref = X.reshape(-1, C).mean(axis=0)
    Xr = X.reshape(-1, C)
    A = np.vstack([Xr.T, np.ones(Xr.shape[0])]).T
    coeffs, *_ = np.linalg.lstsq(A, ref, rcond=None)
    slope = coeffs[:-1]
    intercept = coeffs[-1]
    Xc = (Xr - intercept) / (slope + 1e-8)
    return Xc.reshape(H, W, C)

def zscore(X):
    mu, std = X.mean(), X.std() + 1e-8
    return (X - mu) / std

NORMALIZERS = {
    "snv": snv,
    "msc": msc,
    "zscore": zscore,
    "none": lambda X: X,
}
