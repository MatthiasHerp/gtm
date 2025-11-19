import json
import numpy as np
import pyvinecopulib as pv

# mapping betwen JSON strings and PyVinecopulib enums
FAMILY_MAP = {
    "gaussian": pv.BicopFamily.gaussian,
    "student":  pv.BicopFamily.student,
    "clayton":  pv.BicopFamily.clayton,
    "gumbel":   pv.BicopFamily.gumbel,
    "joe":      pv.BicopFamily.joe,
    "bb1":      pv.BicopFamily.bb1,
    "bb6":      pv.BicopFamily.bb6,
    "bb7":      pv.BicopFamily.bb7,
    "bb8":      pv.BicopFamily.bb8,
    "frank":    pv.BicopFamily.frank,
    "tawn1":    pv.BicopFamily.tawn,
    #"husler_reiss": pv.BicopFamily.husler_reiss
}

def load_copula_configs_from_json(json_path):
    with open(json_path, "r") as f:
        cfgs_raw = json.load(f)

    cfgs = []
    for cfg in cfgs_raw:
        fam = cfg["family"].lower()
        if fam not in FAMILY_MAP:
            raise ValueError(f"Unknown copula family '{fam}' in JSON.")

        params_col = np.array(cfg["params"], ndmin=2).T  # force shape (p,1)

        cfgs.append({
            "name": cfg["name"],
            "family": FAMILY_MAP[fam],
            "params": lambda p=params_col: p.copy(),  # closure
            "rotation": cfg.get("rotation", 0),
        })

    return cfgs
