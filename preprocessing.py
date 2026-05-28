# flowlib/preprocessing.py

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

eps = 1e-6 #avoids infinities in logit. 
band_global = 0.01


def prepare_data(df, subset_size, seed, reflection=False, phi_transform=False): #Phi_transform tries to train on 6 variables (cos phi, sin phi) rather than 5 variables. Although this was attempted it does not work and the implementation may be wrong. 

    df_subset = df.sample(n=subset_size, random_state=seed)

    features = ["m1", "cos1", "m2", "cos2", "phi"]

    X = df_subset[features].values.astype(np.float32)
    #weights = df_subset["weight_detJ"].values.astype(np.float32) #no weights in new file
    #weights = np.ones_like(weights)
    weights = np.ones_like(df_subset["m1"])
    effective_number = weights.sum()

    # Hard-coded physical bounds
    minimum_m1 = 0.649999 #see lab notebook26th feb - tried changing these bounds - now they extend just beyond the physical boundaries, which improves edge behaviour. 
    maximum_m1 = 1.030001
    minimum_cos1 = -1
    maximum_cos1 = 1
    minimum_m2 = 0.649999
    maximum_m2 = 1.030001
    minimum_cos2 = -1
    maximum_cos2 = 1
    minimum_phi = -np.pi
    maximum_phi = np.pi

    if phi_transform == True: #this was the phi transform stuff but it doesn't properly work and may throw errors
        X_phi_cos = np.cos(X[:, 4])
        X_phi_sin = np.sin(X[:, 4])
        X_periodic = np.concatenate(
            [X[:, :4], X_phi_cos[:, None], X_phi_sin[:, None]],
            axis=1
        )
        X = X_periodic
        minimum_phi_cos = -1
        minimum_phi_sin = -1
        maximum_phi_cos = 1
        maximum_phi_sin = 1

    if reflection == True: #this reflection doesn't really work - there may be implementation errors. Either way it is not used
        import hist

        print("reflecting")
        band = band_global

        mask_low = X[:, 0] < (minimum_m1 + band)
        mask_high = X[:, 0] > (maximum_m1 - band)
        mask_lowcos1 = X[:, 1] < (minimum_cos1 + band)
        mask_highcos1 = X[:, 1] > (maximum_cos1 - band)
        mask_lowm2 = X[:, 2] < (minimum_m2 + band)
        mask_highm2 = X[:, 2] > (maximum_m2 - band)
        mask_lowcos2 = X[:, 3] < (minimum_cos2 + band)
        mask_highcos2 = X[:, 3] > (maximum_cos2 - band)
        mask_lowphi = X[:, 4] < (minimum_phi + band)
        mask_highphi = X[:, 4] > (maximum_phi - band)

        n_bins = 50
        hbeforeref = hist.Hist.new.Reg(
            n_bins,
            minimum_m1 - band,
            maximum_m1 + band,
            name="hbeforeref"
        ).Weight()

        fig, ax = plt.subplots(figsize=(8, 6))
        hbeforeref.fill(hbeforeref=X[:, 0], weight=weights)

        hbeforeref.plot(
            ax=ax,
            label="before ref",
            color="black",
            binticks=False,
            histtype="errorbar",
            yerr=True,
            density=False
        )

        X_reflect_low = X[mask_low].copy()
        X_reflect_high = X[mask_high].copy()
        X_reflect_low[:, 0] = 2 * minimum_m1 - X_reflect_low[:, 0]
        X_reflect_high[:, 0] = 2 * maximum_m1 - X_reflect_high[:, 0]
        weights_reflect_low = weights[mask_low].copy()
        weights_reflect_high = weights[mask_high].copy()

        X_reflect_low_cos1 = X[mask_lowcos1].copy()
        X_reflect_high_cos1 = X[mask_highcos1].copy()
        X_reflect_low_cos1[:, 1] = 2 * minimum_cos1 - X_reflect_low_cos1[:, 1]
        X_reflect_high_cos1[:, 1] = 2 * maximum_cos1 - X_reflect_high_cos1[:, 1]
        weights_reflect_low_cos1 = weights[mask_lowcos1].copy()
        weights_reflect_high_cos1 = weights[mask_highcos1].copy()

        X_reflect_low_m2 = X[mask_lowm2].copy()
        X_reflect_high_m2 = X[mask_highm2].copy()
        X_reflect_low_m2[:, 2] = 2 * minimum_m2 - X_reflect_low_m2[:, 2]
        X_reflect_high_m2[:, 2] = 2 * maximum_m2 - X_reflect_high_m2[:, 2]
        weights_reflect_low_m2 = weights[mask_lowm2].copy()
        weights_reflect_high_m2 = weights[mask_highm2].copy()

        X_reflect_low_cos2 = X[mask_lowcos2].copy()
        X_reflect_high_cos2 = X[mask_highcos2].copy()
        X_reflect_low_cos2[:, 3] = 2 * minimum_cos2 - X_reflect_low_cos2[:, 3]
        X_reflect_high_cos2[:, 3] = 2 * maximum_cos2 - X_reflect_high_cos2[:, 3]
        weights_reflect_low_cos2 = weights[mask_lowcos2].copy()
        weights_reflect_high_cos2 = weights[mask_highcos2].copy()

        X_reflect_low_phi = X[mask_lowphi].copy()
        X_reflect_high_phi = X[mask_highphi].copy()
        X_reflect_low_phi[:, 4] = 2 * minimum_phi - X_reflect_low_phi[:, 4]
        X_reflect_high_phi[:, 4] = 2 * maximum_phi - X_reflect_high_phi[:, 4]
        weights_reflect_low_phi = weights[mask_lowphi].copy()
        weights_reflect_high_phi = weights[mask_highphi].copy()

        X = np.concatenate((
            X,
            X_reflect_low, X_reflect_high,
            X_reflect_low_cos1, X_reflect_high_cos1,
            X_reflect_low_m2, X_reflect_high_m2,
            X_reflect_low_cos2, X_reflect_high_cos2,
            X_reflect_low_phi, X_reflect_high_phi
        ), axis=0)

        weights = np.concatenate((
            weights,
            weights_reflect_low, weights_reflect_high,
            weights_reflect_low_cos1, weights_reflect_high_cos1,
            weights_reflect_low_m2, weights_reflect_high_m2,
            weights_reflect_low_cos2, weights_reflect_high_cos2,
            weights_reflect_low_phi, weights_reflect_high_phi
        ), axis=0)

        print(int(effective_number))

    X_train, X_temp, w_train, w_temp = train_test_split(
        X,
        weights,
        test_size=0.3,
        random_state=seed
    )

    X_val, X_finaltest, w_val, w_finaltest = train_test_split(
        X_temp,
        w_temp,
        test_size=0.5,
        random_state=seed
    )

    if phi_transform == True:
        bounds = (
            minimum_m1, maximum_m1,
            minimum_cos1, maximum_cos1,
            minimum_m2, maximum_m2,
            minimum_cos2, maximum_cos2,
            minimum_phi_cos, maximum_phi_cos,
            minimum_phi_sin, maximum_phi_sin
        )
        features = ["m1", "cos1", "m2", "cos2", "phi_cos", "phi_sin"]

    else:
        bounds = (
            minimum_m1, maximum_m1,
            minimum_cos1, maximum_cos1,
            minimum_m2, maximum_m2,
            minimum_cos2, maximum_cos2,
            minimum_phi, maximum_phi
        )

    X_finaltest_physical = X_finaltest.copy()
    X_val_physical = X_val.copy()

    return features, effective_number, X_train, X_val, X_finaltest, \
           w_train, w_val, w_finaltest, bounds, df_subset, \
           X_finaltest_physical, X_val_physical

def prepare_data_for_gendf(df, subset_size, seed, reflection=False, phi_transform=False):

    df_subset = df.sample(n=subset_size, random_state=seed)

    features = ["m1", "cos1", "m2", "cos2", "phi"]

    X = df_subset[features].values.astype(np.float32)
    #weights = df_subset["weight_detJ"].values.astype(np.float32) #no weights in new file
    #weights = np.ones_like(weights)
    weights = np.ones_like(df_subset["m1"])
    effective_number = weights.sum()

    # Hard-coded physical bounds
    minimum_m1 = df_subset['m1'].min() - 1e-6 #0.649999 #see lab notebook26th feb - tried changing these bounds - now they extend just beyond the physical boundaries, which improves edge behaviour. 
    maximum_m1 = df_subset['m1'].max() + 1e-6 #1.030001
    minimum_cos1 = -1
    maximum_cos1 = 1
    minimum_m2 = df_subset['m2'].min() -1e-6 #0.649999 # 19th May: changing bounds to account for fact that the generated distribution can generate data at the bounds of the logit, which are just outside the data range. Correspondingly, when training on a generated distribution, these bounds must be shifted. This would also work in the function above; but I did not want to change that at this late stage. 
    maximum_m2 = df_subset['m2'].max() + 1e-6 #1.030001
    minimum_cos2 = -1
    maximum_cos2 = 1
    minimum_phi = -np.pi
    maximum_phi = np.pi

    if phi_transform == True: #this was the phi transform stuff but it doesn't properly work and may throw errors
        X_phi_cos = np.cos(X[:, 4])
        X_phi_sin = np.sin(X[:, 4])
        X_periodic = np.concatenate(
            [X[:, :4], X_phi_cos[:, None], X_phi_sin[:, None]],
            axis=1
        )
        X = X_periodic
        minimum_phi_cos = -1
        minimum_phi_sin = -1
        maximum_phi_cos = 1
        maximum_phi_sin = 1

    if reflection == True: #this reflection doesn't really work - there may be implementation errors. Either way it is not used
        import hist

        print("reflecting")
        band = band_global

        mask_low = X[:, 0] < (minimum_m1 + band)
        mask_high = X[:, 0] > (maximum_m1 - band)
        mask_lowcos1 = X[:, 1] < (minimum_cos1 + band)
        mask_highcos1 = X[:, 1] > (maximum_cos1 - band)
        mask_lowm2 = X[:, 2] < (minimum_m2 + band)
        mask_highm2 = X[:, 2] > (maximum_m2 - band)
        mask_lowcos2 = X[:, 3] < (minimum_cos2 + band)
        mask_highcos2 = X[:, 3] > (maximum_cos2 - band)
        mask_lowphi = X[:, 4] < (minimum_phi + band)
        mask_highphi = X[:, 4] > (maximum_phi - band)

        n_bins = 50
        hbeforeref = hist.Hist.new.Reg(
            n_bins,
            minimum_m1 - band,
            maximum_m1 + band,
            name="hbeforeref"
        ).Weight()

        fig, ax = plt.subplots(figsize=(8, 6))
        hbeforeref.fill(hbeforeref=X[:, 0], weight=weights)

        hbeforeref.plot(
            ax=ax,
            label="before ref",
            color="black",
            binticks=False,
            histtype="errorbar",
            yerr=True,
            density=False
        )

        X_reflect_low = X[mask_low].copy()
        X_reflect_high = X[mask_high].copy()
        X_reflect_low[:, 0] = 2 * minimum_m1 - X_reflect_low[:, 0]
        X_reflect_high[:, 0] = 2 * maximum_m1 - X_reflect_high[:, 0]
        weights_reflect_low = weights[mask_low].copy()
        weights_reflect_high = weights[mask_high].copy()

        X_reflect_low_cos1 = X[mask_lowcos1].copy()
        X_reflect_high_cos1 = X[mask_highcos1].copy()
        X_reflect_low_cos1[:, 1] = 2 * minimum_cos1 - X_reflect_low_cos1[:, 1]
        X_reflect_high_cos1[:, 1] = 2 * maximum_cos1 - X_reflect_high_cos1[:, 1]
        weights_reflect_low_cos1 = weights[mask_lowcos1].copy()
        weights_reflect_high_cos1 = weights[mask_highcos1].copy()

        X_reflect_low_m2 = X[mask_lowm2].copy()
        X_reflect_high_m2 = X[mask_highm2].copy()
        X_reflect_low_m2[:, 2] = 2 * minimum_m2 - X_reflect_low_m2[:, 2]
        X_reflect_high_m2[:, 2] = 2 * maximum_m2 - X_reflect_high_m2[:, 2]
        weights_reflect_low_m2 = weights[mask_lowm2].copy()
        weights_reflect_high_m2 = weights[mask_highm2].copy()

        X_reflect_low_cos2 = X[mask_lowcos2].copy()
        X_reflect_high_cos2 = X[mask_highcos2].copy()
        X_reflect_low_cos2[:, 3] = 2 * minimum_cos2 - X_reflect_low_cos2[:, 3]
        X_reflect_high_cos2[:, 3] = 2 * maximum_cos2 - X_reflect_high_cos2[:, 3]
        weights_reflect_low_cos2 = weights[mask_lowcos2].copy()
        weights_reflect_high_cos2 = weights[mask_highcos2].copy()

        X_reflect_low_phi = X[mask_lowphi].copy()
        X_reflect_high_phi = X[mask_highphi].copy()
        X_reflect_low_phi[:, 4] = 2 * minimum_phi - X_reflect_low_phi[:, 4]
        X_reflect_high_phi[:, 4] = 2 * maximum_phi - X_reflect_high_phi[:, 4]
        weights_reflect_low_phi = weights[mask_lowphi].copy()
        weights_reflect_high_phi = weights[mask_highphi].copy()

        X = np.concatenate((
            X,
            X_reflect_low, X_reflect_high,
            X_reflect_low_cos1, X_reflect_high_cos1,
            X_reflect_low_m2, X_reflect_high_m2,
            X_reflect_low_cos2, X_reflect_high_cos2,
            X_reflect_low_phi, X_reflect_high_phi
        ), axis=0)

        weights = np.concatenate((
            weights,
            weights_reflect_low, weights_reflect_high,
            weights_reflect_low_cos1, weights_reflect_high_cos1,
            weights_reflect_low_m2, weights_reflect_high_m2,
            weights_reflect_low_cos2, weights_reflect_high_cos2,
            weights_reflect_low_phi, weights_reflect_high_phi
        ), axis=0)

        print(int(effective_number))

    X_train, X_temp, w_train, w_temp = train_test_split(
        X,
        weights,
        test_size=0.3,
        random_state=seed
    )

    X_val, X_finaltest, w_val, w_finaltest = train_test_split(
        X_temp,
        w_temp,
        test_size=0.5,
        random_state=seed
    )

    if phi_transform == True:
        bounds = (
            minimum_m1, maximum_m1,
            minimum_cos1, maximum_cos1,
            minimum_m2, maximum_m2,
            minimum_cos2, maximum_cos2,
            minimum_phi_cos, maximum_phi_cos,
            minimum_phi_sin, maximum_phi_sin
        )
        features = ["m1", "cos1", "m2", "cos2", "phi_cos", "phi_sin"]

    else:
        bounds = (
            minimum_m1, maximum_m1,
            minimum_cos1, maximum_cos1,
            minimum_m2, maximum_m2,
            minimum_cos2, maximum_cos2,
            minimum_phi, maximum_phi
        )

    X_finaltest_physical = X_finaltest.copy()
    X_val_physical = X_val.copy()

    return features, effective_number, X_train, X_val, X_finaltest, \
           w_train, w_val, w_finaltest, bounds, df_subset, \
           X_finaltest_physical, X_val_physical


def prepare_data_with_weights_for_bdt(df, subset_size, seed, reflection=False, phi_transform=False):

    df_subset = df.sample(n=subset_size, random_state=seed)

    features = ["m1", "cos1", "m2", "cos2", "phi"]

    X = df_subset[features].values.astype(np.float32)
    weights = df_subset["Weight"].values.astype(np.float32) #works fine as the calls to this function add a "Weight" column
    effective_number = weights.sum()

    minimum_m1 = 0.649999
    maximum_m1 = 1.030001
    minimum_cos1 = -1
    maximum_cos1 = 1
    minimum_m2 = 0.649999
    maximum_m2 = 1.030001
    minimum_cos2 = -1
    maximum_cos2 = 1
    minimum_phi = -np.pi
    maximum_phi = np.pi

    X_train, X_temp, w_train, w_temp = train_test_split(
        X,
        weights,
        test_size=0.3,
        random_state=seed
    )

    X_val, X_finaltest, w_val, w_finaltest = train_test_split(
        X_temp,
        w_temp,
        test_size=0.5,
        random_state=seed
    )

    bounds = (
        minimum_m1, maximum_m1,
        minimum_cos1, maximum_cos1,
        minimum_m2, maximum_m2,
        minimum_cos2, maximum_cos2,
        minimum_phi, maximum_phi
    )

    X_finaltest_physical = X_finaltest.copy()
    X_val_physical = X_val.copy()

    return features, effective_number, X_train, X_val, X_finaltest, \
           w_train, w_val, w_finaltest, bounds, df_subset, \
           X_finaltest_physical, X_val_physical


def bounded_to_unbounded(column, min_value, max_value, use_tanh=False):
    if use_tanh == False:
        norm = (column - min_value) / (max_value - min_value)
        norm = np.clip(norm, eps, 1 - eps)
        return np.log(norm / (1 - norm))
    # Scales to [-1, 1] first, then applies arctanh
    norm = (2 * (column - min_value) / (max_value - min_value)) - 1
    norm = np.clip(norm, -1 + eps, 1 - eps)
    return np.arctanh(norm)


def unbounded_to_bounded(column, min_value, max_value, use_tanh=False):
    if use_tanh == False:
        sigmoid = 1 / (1 + np.exp(-column))
        return sigmoid * (max_value - min_value) + min_value
    # inverse of arctanh/tanh scaling
    tanh_val = np.tanh(column)
    return (tanh_val + 1) * 0.5 * (max_value - min_value) + min_value


def logit_jacobian(x, min_val, max_val, use_tanh=False):
    if use_tanh == False:
        #x_clamped = torch.clamp(x, min_val + eps, max_val - eps) #corrected this to match bounded_to_unbounded
        x_clamped = torch.clamp(x, #clips in normalised units 
            min_val + eps * (max_val - min_val),
            max_val - eps * (max_val - min_val)
            )
        return torch.log(torch.tensor(max_val - min_val, dtype=torch.float32, device=x.device)) \
               - torch.log(x_clamped - min_val) \
               - torch.log(max_val - x_clamped)
    # For tanh scaling: derivative of arctanh(2*(x-min)/(max-min)-1) #I don't use det J anyway
    scale = 2 / (max_val - min_val)
    x_scaled = torch.clamp(scale * (x - min_val) - 1, -1 + eps, 1 - eps)
    return -torch.log(1 - x_scaled**2) + torch.log(torch.tensor(scale, dtype=torch.float32, device=x.device))


def logit_normalise(X, bounds, use_tanh=False, phi_transform=False):

    if phi_transform == True:
        (min_m1, max_m1,
         min_cos1, max_cos1,
         min_m2, max_m2,
         min_cos2, max_cos2,
         min_phi_cos, max_phi_cos,
         min_phi_sin, max_phi_sin) = bounds
    else:
        (min_m1, max_m1,
         min_cos1, max_cos1,
         min_m2, max_m2,
         min_cos2, max_cos2,
         min_phi, max_phi) = bounds

    X_norm = np.zeros_like(X)

    X_norm[:, 0] = bounded_to_unbounded(X[:, 0], min_m1, max_m1, use_tanh)
    X_norm[:, 1] = bounded_to_unbounded(X[:, 1], min_cos1, max_cos1, use_tanh)
    X_norm[:, 2] = bounded_to_unbounded(X[:, 2], min_m2, max_m2, use_tanh)
    X_norm[:, 3] = bounded_to_unbounded(X[:, 3], min_cos2, max_cos2, use_tanh)

    if phi_transform == True:
        X_norm[:, 4] = bounded_to_unbounded(X[:, 4], min_phi_cos, max_phi_cos, use_tanh)
        X_norm[:, 5] = bounded_to_unbounded(X[:, 5], min_phi_sin, max_phi_sin, use_tanh)
    else:
        X_norm[:, 4] = bounded_to_unbounded(X[:, 4], min_phi, max_phi, use_tanh)

    return X_norm


def convert_to_tensors(X_train, X_val, X_finaltest,
                       w_train, w_val, w_finaltest,
                       device):

    x_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    x_finaltest = torch.tensor(X_finaltest, dtype=torch.float32).to(device)

    w_train = torch.tensor(w_train, dtype=torch.float32).to(device)
    w_val = torch.tensor(w_val, dtype=torch.float32).to(device)
    w_finaltest = torch.tensor(w_finaltest, dtype=torch.float32).to(device)

    return x_train, x_val, x_finaltest, w_train, w_val, w_finaltest


def Poisson_Distribute_eventlevel(data, seed):
    rng = np.random.default_rng(seed)
    return rng.poisson(1, size=len(data)).astype(float)

def Poisson_Distribute(data, seed): #Callum James White's function for Poisson noise for the BDT test - see lab notebook. It adds a lot less noise than the function defined above.

    np.random.seed(seed)
    poisson_weights = np.random.poisson(data.shape[0], size = data.shape[0]) / data.shape[0]

    return poisson_weights