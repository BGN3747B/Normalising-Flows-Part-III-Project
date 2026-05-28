# flowlib/scalers.py

import numpy as np
import torch


class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean


class StandardScalerTorch: #Standard scaler for the flow: needed to normalise variable scales. Non-torch and torch ones defined for ease of implementation later

    def fit(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean


class WhiteningScaler: 

    def __init__(self, mode="zca", eps=1e-5):
        self.mode = mode
        self.eps = eps
        self.mean = None
        self.W = None
        self.W_inv = None

    def fit(self, X):

        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean

        cov = Xc.T @ Xc / Xc.shape[0]

        U, S, Vt = np.linalg.svd(cov)

        if self.mode == "pca":
            self.W = (U / np.sqrt(S + self.eps)).T
            self.W_inv = (U * np.sqrt(S + self.eps)).T #constructs a PCA whitening matrix, by rotating the data into the principal-component basis and rescaling each principal component to unit variance. I don't use this. 

        elif self.mode == "zca":
            self.W = U @ np.diag(1.0 / np.sqrt(S + self.eps)) @ U.T # decorrelates and rescales the variables, but keeps the transformed coordinates as close as possible to the original coordinate system.
            self.W_inv = U @ np.diag(np.sqrt(S + self.eps)) @ U.T # maps whitened data back to the original centred co-ordinates. 

        else:
            raise ValueError("mode must be 'pca' or 'zca'")

        return self

    def transform(self, X):
        Xc = X - self.mean
        return Xc @ self.W.T

    def inverse_transform(self, X):
        return X @ self.W_inv.T + self.mean


class WhiteningScalerTorch:

    def __init__(self, mode="zca", eps=1e-5):
        self.mode = mode
        self.eps = eps
        self.mean = None
        self.W = None
        self.W_inv = None

    def fit(self, X: torch.Tensor):
        self.mean = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean

        cov = Xc.T @ Xc / Xc.shape[0]

        U, S, V = torch.linalg.svd(cov, full_matrices=False)

        if self.mode == "pca":
            self.W = (U / torch.sqrt(S + self.eps)).T
            self.W_inv = (U * torch.sqrt(S + self.eps)).T

        elif self.mode == "zca":
            self.W = U @ torch.diag(1.0 / torch.sqrt(S + self.eps)) @ U.T
            self.W_inv = U @ torch.diag(torch.sqrt(S + self.eps)) @ U.T

        else:
            raise ValueError("mode must be 'pca' or 'zca'")

        return self

    def transform(self, X: torch.Tensor):
        Xc = X - self.mean
        return Xc @ self.W.T

    def fit_transform(self, X: torch.Tensor):
        return self.fit(X).transform(X)

    def inverse_transform(self, X: torch.Tensor):
        return X @ self.W_inv.T + self.mean