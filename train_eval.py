# flowlib/train_eval.py

import time
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from .model import build_flow
from .preprocessing import unbounded_to_bounded
from .uniform import unit_interval_to_bounded
from .seed import set_seed

def train_flow(flow, optimizer, scheduler,
               x_train, w_train,
               x_val, w_val,
               bounds,
               n_epochs=50,
               batch_size=4096,
               device=None,
               plot=True):

    if device is None:
        device = x_train.device

    N = len(x_train)
    train_losses = []
    val_losses = []
    peak_mem_total = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        flow.train()

        perm = torch.randperm(N, device=x_train.device) #so that data are processed in a random order.
        total_train_loss = 0.0
        total_train_weight = 0.0

        for i in range(0, N, batch_size):
 
            index = perm[i:i+batch_size] #considers the case where the last batch is smaller than the batch size.
            xbatch = x_train[index] #constructs batches
            wbatch = w_train[index]
            logJ = 0# sum(
                #logit_jacobian(xbatch[:,k], bounds[2*k], bounds[2*k+1])
                #for k in range(5)
            #) #reintroduced 4th march - removed 5th march. The loss function doesn't need this and this implementation is wrong anyway- see lab notebook)

            weight_sum = wbatch.sum()

            if weight_sum <= 0: #guard against zero weights
                continue

            optimizer.zero_grad()

            log_prob = flow.log_prob(xbatch)
            loss = -(wbatch * (log_prob+logJ)).sum() / weight_sum #computes weighted NLL
            #if epoch == 0 and i < 5 * batch_size: #investigation of why epoch zero sometimes gave huge training NLL - consequence of a bad batch
                #print("batch start:", i)
                #print("loss:", loss.item())
                #print("log_prob mean:", log_prob.mean().item())
                #print("log_prob min:", log_prob.min().item())
                #print("log_prob max:", log_prob.max().item())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * weight_sum.item()
            total_train_weight += weight_sum.item()

        scheduler.step()

        if total_train_weight > 0:
            average_train_loss = total_train_loss / total_train_weight
        else:
            average_train_loss = np.nan

        flow.eval()
        with torch.no_grad():
            val_weight_sum = w_val.sum()
            logJ_val = 0#sum(
                #logit_jacobian(x_val[:,k], bounds[2*k], bounds[2*k+1])
                #for k in range(5)
            #) #reintroduced for checking/testing, 4th march
            val_log_prob = flow.log_prob(x_val)
            val_loss = -(w_val * (val_log_prob +logJ_val)).sum() / val_weight_sum

        train_losses.append(average_train_loss)
        val_losses.append(val_loss.item())

        if device.type == "cuda":
            peak_mem_epoch = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_mem_total = max(peak_mem_total, peak_mem_epoch)
        else:
            peak_mem_epoch = 0.0

        print(
            f"Epoch {epoch}: Train NLL: {average_train_loss:.4f}, "
            f"Val NLL: {val_loss:.4f}, "
            f"Peak GPU memory: {peak_mem_epoch:.2f} MB"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.perf_counter() - start_time

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Total peak GPU memory during training: {peak_mem_total:.2f} MB")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label="Train NLL")
        plt.plot(val_losses, label="Validation NLL")
        plt.xlabel("Epoch")
        plt.ylabel("Negative Log Likelihood")
        plt.legend()
        plt.title("NLL vs Epoch")
        plt.tight_layout()
        plt.show()

    return train_losses, val_losses, training_time, peak_mem_total


def evaluate_and_sample(flow, x_finaltest, w_finaltest,
                        effective_number, bounds, features, scaler,
                        use_tanh=False,
                        phi_transform=False,
                        sample_seed=42,
                        n_sample=None,
                        chunk_size=100000,
                        device=None):

    if device is None:
        device = next(flow.parameters()).device

    flow.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        logJ = 0#sum( 
            #logit_jacobian(x_finaltest[:,k], bounds[2*k], bounds[2*k+1])
            #for k in range(5)
        #)#reintroduced, 4th march 
        # Added section: times density estimation ONLY: in practice this calculation is not that useful as it depends on the test set size so it cannot really be compared to the Legendre method's evaluation of Legendre coefficients. 
        if device.type == "cuda":
            torch.cuda.synchronize()
        density_start_time = time.perf_counter()

        log_prob_finaltest = flow.log_prob(x_finaltest)

        if device.type == "cuda":
            torch.cuda.synchronize()
        density_time = time.perf_counter() - density_start_time
        # End of added section

        final_nll = -(w_finaltest * (flow.log_prob(x_finaltest)+logJ)).sum() / w_finaltest.sum()

        print(f"Final test NLL: {final_nll:.4f}")
        print(f"Density estimation time: {density_time:.4f} seconds")  # ADDED

        torch.manual_seed(sample_seed)

        if n_sample is None:
            n_sample = int(effective_number*2) #twice the size as the base: one wants to minimise errors. 

        sample_chunks = [] #chunked sampling reduces peak memory usage and reduces chance of CUDA OOM errors. 
        n_done = 0

        while n_done < n_sample:
            n_now = min(chunk_size, n_sample - n_done)

            chunk = flow.sample(n_now).cpu().numpy() #moves to cpu and converts to numpy.
            sample_chunks.append(chunk)

            n_done += n_now

        samples = np.concatenate(sample_chunks, axis=0)

        if device.type == "cuda":
            peak_mem_sampling = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            peak_mem_sampling = 0.0
            
        samples = scaler.inverse_transform(samples) #reverses the scaling
    

        for k in range(len(features)):
            samples[:, k] = unbounded_to_bounded(
            samples[:, k],
            bounds[2*k],
            bounds[2*k+1],
            use_tanh=use_tanh #reverses the logit/arctanh
            )
        #samples = unit_interval_to_bounded(samples, bounds)
        #samples[:,0] = np.clip(samples[:,0], 0.65, 1.03) #19th may: trying clipping to the original bounds to bring e.g. the 0.649999 to 0.65 
        #samples[:,2] = np.clip(samples[:,2], 0.65, 1.03)
        if phi_transform == True:# possibly broken as I gave up on this idea
            r = np.sqrt(samples[:, 4]**2 + samples[:, 5]**2)
            r = np.clip(r, 1e-8, None)
            samples[:, 4] /= r
            samples[:, 5] /= r

    if device.type == "cuda":
        torch.cuda.synchronize()

    sampling_time = time.perf_counter() - start_time

    print(f"Peak GPU memory during sampling: {peak_mem_sampling:.2f} MB")
    print(f"Sampling time: {sampling_time:.2f} seconds")
    print(len(samples)) #debugging
    return pd.DataFrame(samples, columns=features), sampling_time, peak_mem_sampling, density_time


def hyperparameter_sweep(sweep_config, 
                         x_train, w_train,
                         x_val, w_val,
                         x_finaltest, w_finaltest,
                         ndim, effective_number,
                         bounds, features,
                         scaler,
                         device,
                         n_epochs=100,
                         use_tanh=False,
                         phi_transform=False,
                         save_prefix="flow_run",
                         training_seed=42,
                         lr=0.001255437468750141,#0.0011884018264173649, #the hyperparameters that are printed are the exact outputs from Optuna. The report reports rounded values. The commented values are from an earlier 50 trial hyperparameter sweep based on the max(ED,0) + six chi2 objective
                         beta2=0.9876911898417069,#0.9971965797064136,   #that scanned a smaller number of MAF layers (2 to 5 layers) and slightly different learning rate range (0.0001 to 0.01). 
                         weight_decay=3.169175618784776e-06):#6.218736992634546e-5):
    """Manual hyperparameter sweep: sweep_config must define layers, hidden features, num_blocks, and batch_size. Other hyperparameters must be modified here."""
    keys = list(sweep_config.keys())
    values = list(sweep_config.values())
    all_combinations = list(itertools.product(*values))

    results = {}

    for combo in all_combinations:
        params = dict(zip(keys, combo))
        set_seed(training_seed)  # This is the training seed used for model initialisation, batch order, dropout, and sampling. For a given architecture, this should be deterministic. 

        print("\n=======================================")
        print("Running configuration:")
        print(params)
        print("=======================================")
        print(f"Training seed: {training_seed}")  # Prints the training seed used 

        num_layers = params.get("num_layers", 7)
        hidden_features = params.get("hidden_features", 64)
        batch_size = params.get("batch_size", 4096)
        num_blocks = params.get("num_blocks", 2)
        use_lu_linear = params.get("use_lu_linear", False)

        flow = build_flow(
            ndim,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            use_lu_linear=use_lu_linear
        ).to(device)

        optimizer = optim.AdamW(
            flow.parameters(),
            betas=(0.9, beta2),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs
        )

        train_losses, val_losses, training_time, peak_mem_training = train_flow(
            flow, optimizer, scheduler,
            x_train, w_train,
            x_val, w_val,
            bounds,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device
        )
        
        tag = str(params).replace(" ", "")
        model_path = f"{save_prefix}_model_{tag}.pt"
        checkpoint_path = f"{save_prefix}_checkpoint_{tag}.pt"
        scaler_path = f"{save_prefix}_scaler_{tag}.pt"

        torch.save(flow.state_dict(), model_path)
        torch.save(scaler, scaler_path)

        checkpoint = { #
            "model_state_dict": {k: v.detach().cpu() for k, v in flow.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "params": params,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "features": features,
            "bounds": bounds,
            "ndim": ndim
            #"scaler_W": scaler.W, #(added 13th March for old data): trying ZCA whitening
            #"scaler_W_inv": scaler.W_inv# ZCA Whitening would need these other two. 
        }

        torch.save(checkpoint, checkpoint_path)

        print(f"Saved model to {model_path}")
        print(f"Saved scaler to {scaler_path}")
        print(f"Saved checkpoint to {checkpoint_path}")

        generated_df, sampling_time, peak_mem_sampling, density_time = evaluate_and_sample(
            flow,
            x_finaltest,
            w_finaltest,
            effective_number,
            bounds,
            features,
            scaler,
            use_tanh=use_tanh,
            phi_transform=phi_transform,
            sample_seed=training_seed, #24th may - introduced this here as well (before it was in separate function version run in a different notebook)
            n_sample=None, #n_sample should be None to have total sample of 2* original data set. 
            chunk_size=100000,
            device=device
        )

        results[str(params)] = {
            "generated_df": generated_df,
            "training_time": training_time,
            "sampling_time": sampling_time,
            "final_val_loss": val_losses[-1],
            "train_curve": train_losses,
            "val_curve": val_losses,
            "peak_mem_sampling": peak_mem_sampling,
            "peak_mem_training": peak_mem_training,
            "checkpoint_path": checkpoint_path,
            "model_path": model_path,
            "scaler_path": scaler_path,
        }

    return results