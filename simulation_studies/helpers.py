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
    bundle_name: str,          # "gtm_freq" or "gtm_bayes"
    model,
    VI=None,
    tau_nodes=None,
    extra_meta: dict | None = None,
    model_init_kwargs: dict | None = None,
    training_output: dict | None = None,
    log_mlflow_model: bool = False,   # <-- NEW: default OFF to avoid MLflow name rules
):
    """
    Logs a reproducible bundle under artifacts/{bundle_name}/...
    """

    meta = {} if extra_meta is None else dict(extra_meta)

    # (A) init kwargs
    if model_init_kwargs is not None:
        _json_save_and_log(
            model_init_kwargs,
            os.path.join(temp_folder, f"{bundle_name}_init.json"),
            artifact_path=f"{bundle_name}/init",
        )

    # (B) ALWAYS log raw state_dict (most robust)
    _torch_save_and_log(
        model.state_dict(),
        os.path.join(temp_folder, f"{bundle_name}_gtm_state_dict.pt"),
        artifact_path=f"{bundle_name}/state",
    )

    # (C) OPTIONAL: log as MLflow model with a FLAT name (no '/')
    if log_mlflow_model:
        # name must not contain / : . % " '
        mlflow.pytorch.log_model(model, artifact_path=f"{bundle_name}/mlflow_model")

    # (D) VI tensors
    if VI is not None:
        vi_state = {
            "vi_mu": VI.mu.detach().cpu(),
            "vi_rho": VI.rho.detach().cpu(),
            "vi_L_unconstrained": VI.L_unconstrained.detach().cpu(),
        }
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

        if hasattr(VI, "state_dict"):
            _torch_save_and_log(
                VI.state_dict(),
                os.path.join(temp_folder, f"{bundle_name}_vi_state_dict.pt"),
                artifact_path=f"{bundle_name}/vi",
            )

    # (E) tau nodes
    if tau_nodes is not None:
        tau_state = {}
        for node_name in ["node4", "node1", "node2"]:
            node = getattr(tau_nodes, node_name, None)
            if node is None:
                continue
            tau_state[node_name] = {
                "mu": node.mu.detach().cpu(),
                "log_sigma": node.log_sigma.detach().cpu(),
            }
        _torch_save_and_log(
            tau_state,
            os.path.join(temp_folder, f"{bundle_name}_tau_nodes_state.pt"),
            artifact_path=f"{bundle_name}/tau_nodes",
        )

    # (F) training output (optional)
    if training_output is not None:
        summary = {}
        for k in ["training_time", "epochs_run", "best_val", "decor_present"]:
            if k in training_output:
                summary[k] = training_output[k]
        _json_save_and_log(
            summary,
            os.path.join(temp_folder, f"{bundle_name}_train_summary.json"),
            artifact_path=f"{bundle_name}/train",
        )
        for k in ["monitor", "loss_history", "val_history"]:
            if k in training_output and training_output[k] is not None:
                _torch_save_and_log(
                    training_output[k],
                    os.path.join(temp_folder, f"{bundle_name}_{k}.pt"),
                    artifact_path=f"{bundle_name}/train",
                )

    # (G) meta
    _json_save_and_log(
        meta,
        os.path.join(temp_folder, f"{bundle_name}_meta.json"),
        artifact_path=f"{bundle_name}/meta",
    )
