FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gfortran git libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gtm
COPY sing_experiments/requirements.docker.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check

# Retain Git metadata: sing_helpers.py verifies this exact upstream commit.
RUN git clone https://github.com/baptistar/SING.git .external/SING \
    && git -C .external/SING checkout --detach 73e1c2ad9a8886854c2ad22c75508ad3cc36df44

COPY sing_experiments/*.py sing_experiments/*.json ./sing_experiments/
COPY demos/pyvinecopulib_simulation_helpers.py ./demos/

# Check imports and upstream compatibility without launching model fitting.
RUN python -c "from sing_experiments.sing_helpers import load_sing; load_sing(); print('SING imports OK')"

CMD ["python", "sing_experiments/run_sing_simulation.py"]
