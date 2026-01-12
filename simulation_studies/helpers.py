import os, json, torch, mlflow

def _torch_save_and_log(obj, local_path, artifact_path=None):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    torch.save(obj, local_path)
    if artifact_path is None:
        mlflow.log_artifact(local_path)
    else:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

def _json_save_and_log(obj, local_path, artifact_path=None):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    if artifact_path is None:
        mlflow.log_artifact(local_path)
    else:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

def log_gtm_and_vi_bundle(
    *,
    temp_folder: str,
    bundle_name: str,          # e.g. "gtm_freq" or "gtm_bayes"
    model,                     # GTM (torch.nn.Module)
    VI=None,                   # your VI_Model (torch.nn.Module-like)
    tau_nodes=None,            # TauPack
    extra_meta: dict | None = None,
):
    """
    Logs:
      - model as mlflow pytorch model
      - model.state_dict() as artifact
      - VI tensors as artifact (mu/rho/L + optional schema)
      - tau_nodes params (mu/log_sigma per node) as artifact
      - metadata json
    """
    # ---------- 1) log the GTM torch model ----------
    mlflow.pytorch.log_model(model, artifact_path=f"{bundle_name}/gtm_model")

    # also log raw state_dict (often handy)
    _torch_save_and_log(
        model.state_dict(),
        os.path.join(temp_folder, f"{bundle_name}_gtm_state_dict.pt"),
        artifact_path=f"{bundle_name}/state",
    )

    meta = {} if extra_meta is None else dict(extra_meta)

    # ---------- 2) log VI state (robust minimal set) ----------
    if VI is not None:
        # if VI is an nn.Module, you *can* also log it as a model,
        # but I still recommend logging minimal tensors for robustness.
        vi_state = {
            "vi_mu": VI.mu.detach().cpu(),
            "vi_rho": VI.rho.detach().cpu(),
            "vi_L_unconstrained": VI.L_unconstrained.detach().cpu(),
        }

        # optional: store schema/blocks if present (only if it exists)
        if hasattr(VI, "_schema"):
            try:
                vi_state["schema_keys"] = [k for k, _ in VI._schema]
            except Exception:
                pass
        if hasattr(VI, "block_sizes"):
            try:
                vi_state["block_sizes"] = list(map(int, VI.block_sizes))
            except Exception:
                pass

        _torch_save_and_log(
            vi_state,
            os.path.join(temp_folder, f"{bundle_name}_vi_state.pt"),
            artifact_path=f"{bundle_name}/vi",
        )

        # If you still want the full VI state_dict too:
        if hasattr(VI, "state_dict"):
            _torch_save_and_log(
                VI.state_dict(),
                os.path.join(temp_folder, f"{bundle_name}_vi_state_dict.pt"),
                artifact_path=f"{bundle_name}/vi",
            )

    # ---------- 3) log tau_nodes state (variational parameters) ----------
    if tau_nodes is not None:
        tau_state = {}
        for node_name in ["node4", "node1", "node2"]:
            node = getattr(tau_nodes, node_name, None)
            if node is None:
                continue
            # GammaTauNode has mu/log_sigma in your code
            tau_state[node_name] = {
                "mu": node.mu.detach().cpu(),
                "log_sigma": node.log_sigma.detach().cpu(),
            }
        _torch_save_and_log(
            tau_state,
            os.path.join(temp_folder, f"{bundle_name}_tau_nodes_state.pt"),
            artifact_path=f"{bundle_name}/tau_nodes",
        )

    # ---------- 4) log metadata ----------
    _json_save_and_log(
        meta,
        os.path.join(temp_folder, f"{bundle_name}_meta.json"),
        artifact_path=f"{bundle_name}/meta",
    )
