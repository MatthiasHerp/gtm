import numpy as np
import torch
import pandas as pd

def kld_tails_assessment(data,
                             loglik_true,
                             loglik_model,
                             thresholds = np.linspace(0, 3, 300)):
        log_lik_diff = loglik_true - loglik_model
        kld_model_any = []
        kld_model_all = []

        for t in thresholds:
            mask_any = (data.abs().numpy() > t).any(1)
            mask_all = (data.abs().numpy() > t).all(1)

            if mask_any.sum() == 0:
                kld_model_any.append(np.nan)
            else:
                kld_model_any.append(torch.mean(log_lik_diff[mask_any]).item())

            if mask_all.sum() == 0:
                kld_model_all.append(np.nan)
            else:
                kld_model_all.append(torch.mean(log_lik_diff[mask_all]).item())

            
        return pd.DataFrame({"kld_any": kld_model_any,
                             "kld_all": kld_model_all,
                             "threshold": thresholds})