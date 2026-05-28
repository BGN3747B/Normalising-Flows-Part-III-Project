# flowlib/metrics.py

import gc
import numpy as np
import torch

from .preprocessing import logit_normalise
from .scalers import StandardScalerTorch, WhiteningScalerTorch


def energy_distance_torch(X, Y):

    N = X.shape[0]
    M = Y.shape[0]

    d_xy = torch.cdist(X, Y, p=2)
    d_xx = torch.cdist(X, X, p=2)
    d_yy = torch.cdist(Y, Y, p=2)

    term1 = 2 * d_xy.mean()
    term2 = d_xx.sum() / (N * (N - 1))
    term3 = d_yy.sum() / (M * (M - 1))

    result = term1 - term2 - term3

    del d_xy, d_xx, d_yy

    return result


def energy_distance_with_permutation_fixed(
    X_np,
    Y_np,
    bounds,
    n_samples=5000,
    n_permutations=100,
    device="cpu",
    use_tanh=False,
    seed=42
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    n = min(len(X_np), len(Y_np), n_samples)

    X_sub = X_np[:n] #uses first n rows
    Y_sub = Y_np[:n]

    X_sub = logit_normalise(X_sub, bounds, use_tanh) 
    Y_sub = logit_normalise(Y_sub, bounds, use_tanh)

    X_sub = torch.tensor(X_sub, dtype=torch.float64, device=device)
    Y_sub = torch.tensor(Y_sub, dtype=torch.float64, device=device)

    X_full_unbounded = logit_normalise(X_np, bounds, use_tanh) # X_np used here so that the standardisation uses the real data distribution, not a small subset.
    X_full_unbounded = torch.tensor(X_full_unbounded, dtype=torch.float64, device=device)

    scaler = StandardScalerTorch() #14th May, different scaler
    #scaler = WhiteningScalerTorch()
    scaler.fit(X_full_unbounded)

    del X_full_unbounded

    X_sub = scaler.transform(X_sub)
    Y_sub = scaler.transform(Y_sub)

    observed = energy_distance_torch(X_sub, Y_sub)
    observed_value = observed.item()
    
    #combining for the permutation test. If the two distributions are indistinguishable, one expects the permutation p-value to be about 0.5, since the energy distance is just as likely to be higher as lower. 
    combined = torch.cat([X_sub, Y_sub], dim=0)

    perm_EDs = []
    count = 0
    
    for _ in range(n_permutations):

        perm = torch.randperm(2 * n, device=device)
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]

        ed_perm = energy_distance_torch(X_perm, Y_perm)
        ed_value = ed_perm.item()

        perm_EDs.append(ed_value)

        if ed_value >= observed_value:
            count += 1

        del perm, X_perm, Y_perm, ed_perm

    p_value = (count + 1) / (n_permutations + 1) #my supervisor's example used count/permutations but this is more robust. 

    del X_sub, Y_sub, combined, observed

    gc.collect()

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    return observed_value, p_value, perm_EDs #If the p-value is <0.05, this suggests that at the 5% significance level, there is sufficient evidence to reject the null hypothesis that the generated and finaltest distributions are the same. 


def sliced_wasserstein_distance_orig( #this implementation has no permutation test. It is not used. 
    X,
    Y,
    n_projections=300,
    n_samples=100000,
    device="cuda",
    seed=42
): 

    torch.manual_seed(seed)

    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)

    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.float32)

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    X = X.to(device)
    Y = Y.to(device)

    n = min(n_samples, X.shape[0], Y.shape[0]) #subsamples to equal size

    idx_x = torch.randperm(X.shape[0], device=device)[:n]
    idx_y = torch.randperm(Y.shape[0], device=device)[:n]

    X = X[idx_x]
    Y = Y[idx_y]

    d = X.shape[1]

    projections = torch.randn(n_projections, d, device=device) #random projection directions 
    projections = projections / projections.norm(dim=1, keepdim=True)

    proj_X = X @ projections.T #projects the data
    proj_Y = Y @ projections.T

    proj_X_sorted, _ = torch.sort(proj_X, dim=0) #sorts the projections
    proj_Y_sorted, _ = torch.sort(proj_Y, dim=0)

    swd = torch.mean(torch.abs(proj_X_sorted - proj_Y_sorted)) #calculates the swd, by averaging the 1D wasserstein distances over the projections

    return swd.item() 


def calc_of_sinkhorn(X_np, Y_np, device="cuda", n_samples=5000, seed=42): #Sinkhorn calculation using GeomLoss package. 

    from geomloss import SamplesLoss

    rng = np.random.default_rng(seed)

    n = min(n_samples, len(X_np), len(Y_np))

    idx1 = rng.choice(len(X_np), n, replace=False)
    idx2 = rng.choice(len(Y_np), n, replace=False)

    x = torch.tensor(X_np[idx1], dtype=torch.float32, device=device)
    y = torch.tensor(Y_np[idx2], dtype=torch.float32, device=device)

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    out = loss_fn(x, y).item()

    del x, y

    return out


def sliced_wasserstein_fixed_projections_torch(X, Y, projections):
    """
    X, Y: torch tensors of shape (n, d)
    projections: torch tensor of shape (n_projections, d)

    It projects, sorts, and calculates the mean absolute difference.
    """

    proj_X = X @ projections.T #projects datasets onto all projection directions
    proj_Y = Y @ projections.T

    proj_X_sorted, _ = torch.sort(proj_X, dim=0) #Sorts because in 1D, Wasserstein distance is computed by pairing sorted samples.
    proj_Y_sorted, _ = torch.sort(proj_Y, dim=0)

    swd = torch.mean(torch.abs(proj_X_sorted - proj_Y_sorted)) #computes SWD by averaging sorted projected samples over all samples and projections. 

    return swd



def sliced_wasserstein_with_permutation(
    X_np,
    Y_np,
    n_samples=100000,
    n_projections=300,
    n_permutations=100,
    device="cuda",
    seed=42):
    """
    This function calculates the SWD and a permutation p-value.

    The SWD uses the same logic as sliced_wasserstein_distance_orig:
        random subsample
        random projection directions
        sorted projected samples
        mean absolute difference

    The permutation test pools the two samples, randomly relabels them,
    and recomputes the SWD using the same projection directions.
    """

    torch.manual_seed(seed)

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    if not torch.is_tensor(X_np):
        X_all = torch.tensor(X_np, dtype=torch.float32, device=device)
    else:
        X_all = X_np.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(Y_np):
        Y_all = torch.tensor(Y_np, dtype=torch.float32, device=device)
    else:
        Y_all = Y_np.to(device=device, dtype=torch.float32)

    n = min(n_samples, X_all.shape[0], Y_all.shape[0])

    # torch.randperm after torch.manual_seed ensures repeatability
    idx_x = torch.randperm(X_all.shape[0], device=device)[:n]
    idx_y = torch.randperm(Y_all.shape[0], device=device)[:n]

    X = X_all[idx_x]
    Y = Y_all[idx_y]

    d = X.shape[1]

    projections = torch.randn(n_projections, d, device=device) #creates projections 
    projections = projections / projections.norm(dim=1, keepdim=True) #normalises

    observed = sliced_wasserstein_fixed_projections_torch(X, Y, projections)
    observed_value = observed.item()

    combined = torch.cat([X, Y], dim=0) #concatenates X and Y for the permutation test

    perm_swds = []
    count = 0

    for _ in range(n_permutations):
        perm = torch.randperm(2 * n, device=device)

        X_perm = combined[perm[:n]] #Creates a new X and Y from the first and second halves of combined. 
        Y_perm = combined[perm[n:]]

        swd_perm = sliced_wasserstein_fixed_projections_torch(
            X_perm,
            Y_perm,
            projections
        )

        swd_value = swd_perm.item()
        perm_swds.append(swd_value)

        if swd_value >= observed_value: #if the observed SWD tends always to be larger than the permutation SWDs, then it suggests that the two distributions are not in fact the same, and the generated distribution is not a good fit. 
            count += 1

        del perm, X_perm, Y_perm, swd_perm

    p_value = (count + 1) / (n_permutations + 1) 

    del X_all, Y_all, X, Y, combined, projections, observed
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return observed_value, p_value, perm_swds


def energy_distance_with_permutation_fixed_rand( #essentially the same as energy distance above but here using a random subsample instead of the first n rows. 
    X_np,
    Y_np,
    bounds,
    n_samples=5000,
    n_permutations=100,
    device="cpu",
    use_tanh=False,
    seed=42
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    n = min(len(X_np), len(Y_np), n_samples) #Random subsample using a fixed seed 

    rng = np.random.default_rng(seed)
    idx_x = rng.choice(len(X_np), size=n, replace=False)
    idx_y = rng.choice(len(Y_np), size=n, replace=False)

    X_sub = X_np[idx_x]
    Y_sub = Y_np[idx_y]

    # LOGIT NORMALISATION: This function works (as above) works in the unbounded space.
    X_sub = logit_normalise(X_sub, bounds, use_tanh)
    Y_sub = logit_normalise(Y_sub, bounds, use_tanh)

    X_sub = torch.tensor(X_sub, dtype=torch.float64, device=device)
    Y_sub = torch.tensor(Y_sub, dtype=torch.float64, device=device)

    X_full_unbounded = logit_normalise(X_np, bounds, use_tanh)
    X_full_unbounded = torch.tensor(X_full_unbounded, dtype=torch.float64, device=device)
    
    #Standard Scaler: uses whole truth distribution, as opposed to just the training sample. 
    scaler = StandardScalerTorch()
    scaler.fit(X_full_unbounded)

    del X_full_unbounded

    X_sub = scaler.transform(X_sub)
    Y_sub = scaler.transform(Y_sub)

    observed = energy_distance_torch(X_sub, Y_sub)
    observed_value = observed.item()

    combined = torch.cat([X_sub, Y_sub], dim=0)

    perm_EDs = []
    count = 0

    for _ in range(n_permutations):

        perm = torch.randperm(2 * n, device=device)
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]

        ed_perm = energy_distance_torch(X_perm, Y_perm)
        ed_value = ed_perm.item()

        perm_EDs.append(ed_value)

        if ed_value >= observed_value:
            count += 1

        del perm, X_perm, Y_perm, ed_perm

    p_value = (count + 1) / (n_permutations + 1)

    del X_sub, Y_sub, combined, observed

    gc.collect()

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    return observed_value, p_value, perm_EDs

