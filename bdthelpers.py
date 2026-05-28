# flowlib/bdt_pipeline.py

import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

from F_BDT_Functions_copy import train_xgb_classifier #importing one of Callum James White's functions

from .preprocessing import (
    prepare_data_with_weights_for_bdt,
    logit_normalise,
    convert_to_tensors,
    unbounded_to_bounded,
    Poisson_Distribute_eventlevel, Poisson_Distribute
)
from .scalers import StandardScaler
from .model import build_flow
from .train_eval import train_flow

def BDT_KFold( #Callum James White's function, suitably modified for my flow pipeline (hence copied and pasted rather than imported )
    dataset_A, dataset_B,
    weight_A, weight_B,
    var_names, seed=5,
    max_depth=3, eta=0.6, subsample=0.8,
    n_estimators=150, n_splits=5, save=''
):
    print("Starting K-Fold BDT comparison...")

    # Label and weight
    dataset_A = dataset_A.copy()
    dataset_B = dataset_B.copy()
    dataset_A["label"] = 1
    dataset_B["label"] = 0
    dataset_A["Weight"] = weight_A
    dataset_B["Weight"] = weight_B


    # Clip negative weights to zero
    # for dataset in [dataset_A, dataset_B]:
    #     dataset['Weight'] = np.clip(dataset['Weight'], a_min=0, a_max=None)

    # Clip variables to [-1, 1] #MY CHANGE FOR FLOW PIPELINE: This is now commented as the flow does not clip to [-1, 1]
    #for var in var_names:
        #dataset_A = dataset_A.loc[(dataset_A[var] >= -1) & (dataset_A[var] <= 1)].copy()
        #dataset_B = dataset_B.loc[(dataset_B[var] >= -1) & (dataset_B[var] <= 1)].copy()

    # Merge datasets
    data = pd.concat([dataset_A, dataset_B], sort=False, ignore_index=True)
    X = data[var_names]
    y = data["label"]
    w = data["Weight"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    aucs, importances = [], []
    tprs, auc_fprs = [], np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 6))
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train, w_test = w.iloc[train_idx], w.iloc[test_idx]

        # Train only on positive-weight events
        # mask_train = w_train > 0
        # X_train_pos = X_train[mask_train]
        # y_train_pos = y_train[mask_train]
        # w_train_pos = w_train[mask_train]
        X_train_pos = X_train
        y_train_pos = y_train
        w_train_pos = w_train

        # if X_train_pos.shape[0] == 0:
        #     print(f"⚠️  Fold {fold_idx+1}: No positive-weight training events! Skipping.")
        #     continue

        # Train model
        model = train_xgb_classifier(
            X_train_pos, X_test, y_train_pos, y_test,
            w_train_pos, w_test,
            name=None, max_depth=max_depth, eta=eta,
            subsample=subsample, n_estimators=n_estimators
        )

        # Ensure correct probability column
        if hasattr(model, "classes_"):
            col_idx = np.where(model.classes_ == 1)[0][0]
        else:
            col_idx = 1

        # --- Compute training AUC ---
        y_train_pred = model.predict_proba(X_train_pos)[:, col_idx]
        train_auc = roc_auc_score(y_train_pos, y_train_pred, sample_weight=w_train_pos)

        # --- Validation prediction ---
        X_test_pred = model.predict_proba(X_test)[:, col_idx]

        # Evaluate only on positive weights
        mask_eval = w_test > 0
        X_test_pred_eval = X_test_pred[mask_eval]
        y_test_eval = y_test[mask_eval]
        w_test_eval = w_test[mask_eval]

        # Compute AUC
        val_auc = roc_auc_score(y_test_eval, X_test_pred_eval, sample_weight=w_test_eval)
        aucs.append(val_auc)
        importances.append(model.feature_importances_)
        print(f"✅ Fold {fold_idx + 1} | Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test_eval, X_test_pred_eval, sample_weight=w_test_eval)
        tpr_interp = np.interp(auc_fprs, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f"Fold {fold_idx + 1} (AUC={val_auc:.3f})")

    # Plot average ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(
        auc_fprs, mean_tpr, color="b",
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        lw=2.5, alpha=0.8
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random", lw=1.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Across K-Folds")
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.tight_layout()
    if save != '':
        plt.savefig(save)

    # Average feature importances
    if len(importances) > 0:
        mean_importance = np.mean(importances, axis=0)
    else:
        mean_importance = np.zeros(len(var_names))

    dataset_A['xgb_output'] = model.predict_proba(dataset_A[var_names].values)[:, col_idx]
    dataset_B['xgb_output'] = model.predict_proba(dataset_B[var_names].values)[:, col_idx]

    print("\n=============================================")
    print(f"Average AUC over {n_splits} folds: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Size Check - Dataset A: {dataset_A.shape[0]}, Dataset B: {dataset_B.shape[0]}")
    print(f"Weight Check - Dataset A: {dataset_A['Weight'].sum()}, Dataset B: {dataset_B['Weight'].sum()}")
    print(f"Mean BDT Score - Dataset A: {dataset_A['xgb_output'].mean():.4f}, Dataset B: {dataset_B['xgb_output'].mean():.4f}")
    print("Average Feature Importance:")
    for name, imp in sorted(zip(var_names, mean_importance), key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.4f}")
    print("=============================================\n")

    return mean_auc, mean_importance, model

def train_and_generate(df, subset_size, seed, device, n_epochs=100, scaler=None, use_tanh=False): #changed to the new thing, 27th March (comment 7th May: by this I think that I meant 20th March as that is where I mention using the same scaler for df1 and df2 when doing the df+noise1 and df+noise2 test)
    """Helper function for the BDT"""
    features, effective_number, X_train, X_val, X_test, \
    w_train, w_val, w_test, bounds, df_subset, _, _ = prepare_data_with_weights_for_bdt(
        df, subset_size, seed, reflection=False
    )

    X_train = logit_normalise(X_train, bounds, use_tanh, phi_transform) #added 27th March - THE PREVIOUS OMISSION WAS A BUG
    X_val = logit_normalise(X_val, bounds, use_tanh, phi_transform)
    X_test = logit_normalise(X_test, bounds, use_tanh, phi_transform)
    
    if scaler==None:
    # Normalisation
        scaler = StandardScaler().fit(X_train) #Only works with Standard scaler for now (comment added 7th May)). The idea here is that I only fit a scaler once. 
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Tensors
    x_train, x_val, x_test, w_train, w_val, w_test = convert_to_tensors(
        X_train, X_val, X_test,
        w_train, w_val, w_test,
        device
    )

    # Builds and trains the flow
    flow = build_flow(ndim=5, hidden_features=429,num_layers=5, num_blocks=2).to(device)

    #optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3) #changed 27th March to the new hyperparams
    optimizer = optim.AdamW(flow.parameters(), betas= (0.9, 0.9971965797064136), lr=0.0011884018264173649, weight_decay= 6.218736992634546e-5)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler= lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_flow(
        flow, optimizer, scheduler,
        x_train, w_train,
        x_val, w_val,
        bounds,
        n_epochs=100, #fixed epochs here
        batch_size= 3469 # keep small for toys
    )

    # Samples 
    generated_df, _ = evaluate_and_sample_for_bdt(
        flow,
        x_test,
        w_test,
        effective_number,
        bounds,
        features,
        scaler
    )

    return generated_df, scaler

def train_and_generatenewhypers(df, subset_size, seed, device, n_epochs=100, scaler=None, use_tanh=False, phi_transform=False): #changed to the new thing, 27th March (comment 7th May: by this I think that I meant 20th March as that is where I mention using the same scaler for df1 and df2 when doing the df+noise1 and df+noise2 test)
    """Helper function for the BDT"""
    features, effective_number, X_train, X_val, X_test, \
    w_train, w_val, w_test, bounds, df_subset, _, _ = prepare_data_with_weights_for_bdt(
        df, subset_size, seed, reflection=False 
    ) #having seed in this argument is sub-optimal, as it means that i cannot actually call it for different seeds (as the same train/val/test split is still required) 

    X_train = logit_normalise(X_train, bounds, use_tanh, phi_transform) #added 27th March - THE PREVIOUS OMISSION WAS A BUG
    X_val = logit_normalise(X_val, bounds, use_tanh, phi_transform)
    X_test = logit_normalise(X_test, bounds, use_tanh, phi_transform)
    
    if scaler==None:
    # Normalisation
        scaler = StandardScaler().fit(X_train) #Only works with Standard scaler for now (comment added 7th May)). The idea here is that I only fit a scaler once. 
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Tensors
    x_train, x_val, x_test, w_train, w_val, w_test = convert_to_tensors(
        X_train, X_val, X_test,
        w_train, w_val, w_test,
        device
    )

    # Builds and trains the flow
    flow = build_flow(ndim=5, hidden_features=478,num_layers=8, num_blocks=3).to(device)

    #optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3) #changed 27th March to the new hyperparams
    optimizer = optim.AdamW(flow.parameters(), betas= (0.9, 0.9876911898417069), lr=0.001255437468750141, weight_decay= 3.169175618784776e-06)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler= lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_flow(
        flow, optimizer, scheduler,
        x_train, w_train,
        x_val, w_val,
        bounds,
        n_epochs=100, #fixed epochs here
        batch_size= 2012 # keep small for toys
    )

    # Samples 
    generated_df, _ = evaluate_and_sample_for_bdt(
        flow,
        x_test,
        w_test,
        effective_number,
        bounds,
        features,
        scaler
    )

    return generated_df, scaler



def evaluate_and_sample_for_bdt(flow, x_finaltest, w_finaltest,
                        effective_number, bounds, features, scaler, reflection=False, use_tanh=False):
    #band = band_global #ignore this, not doing reflection any more
    flow.eval()
    start_time_of_sampling = time.perf_counter()
    with torch.no_grad():

        logJ = 0#sum( 
            #logit_jacobian(x_finaltest[:,k], bounds[2*k], bounds[2*k+1])
            #for k in range(5)
        #)#reintroduced, 4th march 
        if x_finaltest is not None and w_finaltest is not None: #I don't use this
            final_nll = (-w_finaltest *
                         (flow.log_prob(x_finaltest) + logJ)).mean()

            print(f"Final test NLL: {final_nll:.4f}")
        #torch.manual_seed(42) #commented out here because in toy_vs_toy_flow_from_checkpoint I change the seed before calling. 
        samples = flow.sample(int(effective_number * 2)).cpu().numpy() #14th march, changed to *2 as trying to train on full set. 
        # INVERSE STANDARD SCALING 
        samples = scaler.inverse_transform(samples)  #I don't do chunked sampling here (In practice that would be better)

        for k in range(5):
            samples[:,k] = unbounded_to_bounded(
                samples[:,k], bounds[2*k], bounds[2*k+1], use_tanh=use_tanh
            )
        #if reflection==True:
            #print("Sampling and filtering, as reflection turned on")
            #mask = ((samples[:,0] >= bounds[0] + band) & (samples[:,0] <=bounds[1] - band) & (samples[:,1] >= bounds[2]+band) & (samples[:,1] <=bounds[3]-band) & (samples[:,2] >= bounds[4]+ band) & (samples[:,1] <=bounds[5] - band) + (samples[:,3] >= bounds[6] + band) & (samples[:,3] <=bounds[7] - band) & (samples[:,4] >= bounds[8] + band) & (samples[:,1] <=bounds[9] - band))
            #samples = samples[mask] # for now just m1 reflection, cos1, m2, cos2, phi unchanged

    end_time_of_sampling = time.perf_counter()
    sampling_time = end_time_of_sampling - start_time_of_sampling
    print(f' Sampling time: {sampling_time} seconds')
    #
    return pd.DataFrame(samples, columns=features), sampling_time