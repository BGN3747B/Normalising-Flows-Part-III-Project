import gc
import numpy as np
from hyppo.ksample import Energy, MMD
from sklearn.metrics import pairwise_distances

#These functions use Hyppo ED and MMD implementations. 
def _clean_subsample_raw(
    X_np,
    Y_np,
    n_samples=5000,
    seed=42
):
    """
    Randomly subsamples equal-sized X/Y samples with no logit transform.
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(X_np, dtype=np.float64)
    Y = np.asarray(Y_np, dtype=np.float64)

    X = X[np.isfinite(X).all(axis=1)]
    Y = Y[np.isfinite(Y).all(axis=1)]

    n = min(n_samples, len(X), len(Y))

    idx_x = rng.choice(len(X), n, replace=False)
    idx_y = rng.choice(len(Y), n, replace=False)

    return X[idx_x], Y[idx_y]


def _standardise_using_data(X, Y, eps=1e-12):
    """
    Standardisation using the Data sample only, which puts variables
    on comparable numerical scales for Euclidean/kernel metrics.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.where(std < eps, 1.0, std)

    return (X - mean) / std, (Y - mean) / std


def _median_heuristic_gamma(X, Y, max_points=2000, seed=42):
    """
    Median heuristic for the RBF/Gaussian kernel bandwidth. 
    The purpose is to return a bandwidth = 1/(2 sigma^2)  where sigma is the median distance between points in a pooled sample Z of X and Y, thus making the MMD sufficiently sensitive. 

    RBF kernel convention:
        k(x, y) = exp(-gamma * ||x-y||^2)

    If sigma is the median pairwise distance, then:
        gamma = 1 / (2 sigma^2)
    """
    rng = np.random.default_rng(seed)

    Z = np.vstack([X, Y]) #combines X and Y. 

    if len(Z) > max_points: #takes a subsample as the generated distribution is generally very large
        idx = rng.choice(len(Z), max_points, replace=False)
        Z = Z[idx]

    dists = pairwise_distances(Z, metric="euclidean") #computes pairwise distances between points in the pooled sample.
    upper = dists[np.triu_indices_from(dists, k=1)] #uses the upper triangular part of the matrix only, avoiding self-distances and duplicate distances.
    upper = upper[upper > 0] #removes other zero distances (identical points)

    if len(upper) == 0:
        return 1.0 #returns one if all points identical

    sigma = np.median(upper) 

    return 1.0 / (2.0 * sigma**2) #returns the bandwidth. 

def energy_hyppo_raw(
    X_np,
    Y_np,
    n_samples=5000,
    n_permutations=100,
    seed=42,
    standardise=True,
    auto=False,
    workers=1,
    bias=False
):
    """
    Energy two-sample test using hyppo.

    No logit/sigmoid transforms.

    Returns:
        stat, p_value
    """
    X, Y = _clean_subsample_raw(
        X_np,
        Y_np,
        n_samples=n_samples,
        seed=seed
    )

    if standardise:
        X, Y = _standardise_using_data(X, Y)

    test = Energy(
        compute_distance="euclidean",
        bias=bias
    )

    stat, p_value = test.test(
        X,
        Y,
        reps=n_permutations,
        auto=auto,
        workers=workers,
        random_state=seed
    )

    gc.collect()

    return stat, p_value

def mmd_hyppo_raw(
    X_np,
    Y_np,
    n_samples=5000,
    n_permutations=100,
    seed=42,
    standardise=True,
    auto=False,
    workers=1,
    bias=False,
    compute_kernel="gaussian",
    gamma="median"
):
    """
    MMD two-sample test using hyppo.

    No logit/sigmoid transforms.

    Parameters:
        compute_kernel:
            Usually "gaussian" or "rbf".
        gamma:
            "median"  uses median heuristic
            None      uses sklearn/hyppo default
            float     uses supplied gamma
    """
    X, Y = _clean_subsample_raw(
        X_np,
        Y_np,
        n_samples=n_samples,
        seed=seed
    )

    if standardise:
        X, Y = _standardise_using_data(X, Y)

    kernel_kwargs = {}

    if gamma == "median":
        kernel_kwargs["gamma"] = _median_heuristic_gamma(
            X,
            Y,
            seed=seed
        )
    elif gamma is not None:
        kernel_kwargs["gamma"] = float(gamma)

    test = MMD(
        compute_kernel=compute_kernel,
        bias=bias,
        **kernel_kwargs
    )

    stat, p_value = test.test(
        X,
        Y,
        reps=n_permutations,
        auto=auto,
        workers=workers,
        random_state=seed
    )

    gc.collect()

    return stat, p_value, kernel_kwargs