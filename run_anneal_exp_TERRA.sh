#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --job-name=Anneal_sweep_TERRA_NR
#SBATCH --output=Anneal_sweep_TERRA_NR.out
#SBATCH --error=Anneal_sweep_error_TERRA_NR.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:t4:4
#SBATCH --time=4-00:00:00
#SBATCH --mem=32Gb

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index torch torchvision
pip3 install --no-index tqdm

cd $HOME/GitRepos/DomainBed/

python3 -m domainbed.scripts.train\
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/anneal_experiment/results/TERRA_ERM/1/\
       --algorithm ERM \
       --dataset TerraIncognita \
       --steps 300 \
       --seed 1 \
       --trial_seed 1
       
python3 -m domainbed.scripts.train\
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/anneal_experiment/results/TERRA_ERM/2/\
       --algorithm ERM \
       --dataset TerraIncognita \
       --steps 300 \
       --seed 2 \
       --trial_seed 2
       
python3 -m domainbed.scripts.train\
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/anneal_experiment/results/TERRA_ERM/3/\
       --algorithm ERM \
       --dataset TerraIncognita \
       --steps 300 \
       --seed 3 \
       --trial_seed 3

python3 -m domainbed.scripts.anneal_sweep launch\
       --algorithm IRM VREx \
       --dataset TerraIncognita \
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/anneal_experiment/results/TERRA_NR/ \
       --command_launcher multi_gpu \
       --skip_confirmation \
       --steps 300 \
       --n_trials 2 \
       --n_anneal 20


