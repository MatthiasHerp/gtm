from collections import deque

class Trackers:
    # ------------------- monitoring containers
    
    monitor = {
        "epoch": [],

        # taus (hyper + VI means + EB targets)
        "tau4": [], "tau1": [], "tau2": [],
        "tau4_mean": [], "tau1_mean": [], "tau2_mean": [],
        "tau4_target": [], "tau1_target": [], "tau2_target": [],

        # τ·E[qf] monitors
        "tau4_Eqf": [], "tau1_Eqf1": [], "tau2_Eqf2": [],
        "tau4_target_Eqf": [], "tau1_target_Eqf1": [], "tau2_target_Eqf2": [],

        # θ statistics (μ and σ)
        "mu_norm": [],          # ||μ||₂
        "mu_mean": [],          # mean(μ)
        "mu_std": [],           # std(μ)
        "sigma_mean": [],       # mean(σ)
        "sigma_min": [],        # min(σ)
        "sigma_max": [],        # max(σ)

        # ELBO / likelihood
        "train_loss": [],       # per-obs loss = -ELBO/obs
        "elbo_per_obs": [],     # ELBO/obs
        "val_ELPD": [],         # None if no val
    }


class _ELBOConvergence:
    def __init__(self, use_ema=True, window_size=5, tol=1e-5, beta=0.9, min_epochs=10):
        self.use_ema = use_ema
        self.window = deque(maxlen=window_size)
        self.tol = float(tol)
        self.beta = float(beta)
        self.min_epochs = int(min_epochs)
        self._ema = None
        self._past = None
        self._epochs = 0

    def update(self, elbo_per_obs: float) -> tuple[bool, float]:
        """Returns (converged?, smoothed_value)."""
        self._epochs += 1
        if self.use_ema:
            if self._ema is None:
                self._ema = elbo_per_obs
            else:
                self._ema = self.beta * self._ema + (1.0 - self.beta) * elbo_per_obs
            smoothed = self._ema
        else:
            self.window.append(elbo_per_obs)
            smoothed = sum(self.window) / len(self.window)

        if self._epochs < self.min_epochs:
            self._past = smoothed
            return (False, smoothed)

        # absolute change; switch to relative if you prefer
        delta = abs(smoothed - (self._past if self._past is not None else smoothed))
        self._past = smoothed
        return (delta < self.tol, smoothed)

