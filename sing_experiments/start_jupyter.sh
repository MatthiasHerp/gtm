#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export JUPYTER_DATA_DIR="$PROJECT_ROOT/.jupyter-sing/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/.jupyter-sing/runtime"
export JUPYTER_CONFIG_DIR="$PROJECT_ROOT/.jupyter-sing/config"
export IPYTHONDIR="$PROJECT_ROOT/.jupyter-sing/ipython"
export MPLCONFIGDIR="$PROJECT_ROOT/.jupyter-sing/matplotlib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
mkdir -p "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$JUPYTER_CONFIG_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv-sing/bin/jupyter" lab sing_experiments/sing_rvine_10D.ipynb "$@"
