## Docker Basics
### 1. Image: 
- Dockerfile that describes how the container (environment) in which the code is run is configured

### 2. Docker Container: 
- The Container needs to be built from the Image once. 
- In essence the instructions show what to download. 
- Rebuild if you ever reconfigure the container Image (instructions).
- Build with the following code which creates a container named "container_name":
docker build -t container_name -f simulation_studies/Dockerfile_gtm .
- build the docker container in scratch by navigating there with "cd /scratch" and building there

### 3. Running python scripts in the container:
- use the following command, adjust to your setting: 
docker run -it --rm --gpus \"device=1\" -v /scratch/fcc/gtm/:/mnt container_name python3 tests/run_experiment.py

- notice:
    - --gpus \"device=1\" --> we run with gpus on the gpu number 1
    - -v /scratch/fcc/gtm/:/mnt --> we mount a directory to which the code in the container has access, we name it mnt for mount within the container so code in the container should reference mnt
    - notice we mount from scratch, scratch is a faster directory always do this, more later
    - container_name --> need to state the container to use
    - python3 --> runs a python script with python3
    - tests/run_experiment.py -> relative directory from the mount to the file to run

### 4. Use Scratch
- scratch is a faster to acess and write directory on the sever
- push everything to scratch, mount it and compute there, then pull scratch back after finishing computations
- pushing stuff to scratch via rsync command:
rsync -av --delete gtm/* /scratch/fcc/gtm/
- pulling stuff from scratch (only need mlflow in our case):
rsync -av /scratch/mhe/gtm/mlruns/*  /sybig/home/mhe/gtm/mlruns

## 5. Push and Pull Server to local
- either via github syncing or with rsync commands
- rsync push (run locally):
rsync -arvz --delete -e 'ssh deepthought' /Users/matthiasherp/Desktop/phd_github_repositories/gtm/ :/sybig/home/mhe/gtm/.
- rsync pull (run locally):
rsync -arvz -e 'ssh hydra' :/sybig/home/mhe/mctm_pytorch/mlruns .

## 6. Server housekeeping
- erst htop checken (cpu)
- erst nvtop checken welche grafikkarten benutzt sind (maximal 4 stück benutzen)
- verlasse nvtop und htop mit "Q" taste
- wenn rechnen dann in den matrix/element chat schreiben: rechne auf hydra mit ... gpus für ca. ... stunden
- rechne am besten die "docker run ...." in screens (wie tmux):
    - "screen" to start
    - "screen -ls" to see screens active
    - "screen -r ..." wieder rein (siehst die nummer in ls vorne)
    - "screen killall" machst alle aus
