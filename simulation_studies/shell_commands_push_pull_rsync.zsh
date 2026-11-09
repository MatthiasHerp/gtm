# pushing local updates to gtm code base do via github to remote by commit and push
# then pull from remote to server, easier and more reproducible then via rsync

# push everthing from backup on the server to the scratch on the server
# deletes stuff / overwrites
rsync -av --delete gtm/* /scratch/mhe/gtm/

# pull demos or mlruns results from computing from scratch to the backup on the server
# Does not delete stuff
rsync -av /scratch/mhe/gtm/demos/*  gtm/demos/
rsync -av /scratch/mhe/gtm/mlruns/*  gtm/mlruns/

# pulling mlruns to local, by not using github
# run locally
# run in gtm folder root, uses the ssh hydra config and keys
rsync -arvz -e ssh hydra:/sybig/home/mhe/gtm/mlruns . 


rsync -av --delete gtm/gtm/* /scratch/mhe/gtm/gtm/
rsync -av --delete /sybig/home/mhe/gtm/dragon_application/* /scratch/mhe/gtm/dragon_application/
rsync -av --delete /sybig/home/mhe/gtm/simulation_studies/run_experiment_nfm.py /scratch/mhe/gtm/simulation_studies/
rsync -av --delete /sybig/home/mhe/gtm/simulation_studies/run_experiment.py /scratch/mhe/gtm/simulation_studies/

rsync -av /sybig/home/mhe/gtm/simulation_studies/run_files_paper/* /scratch/mhe/gtm/simulation_studies/run_files_paper/
rsync -av /sybig/home/mhe/gtm/simulation_studies/run_files_paper/* /scratch/mhe/gtm/simulation_studies/run_files_paper/