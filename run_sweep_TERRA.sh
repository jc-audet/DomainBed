#!/bin/bash
#SBATCH --job-name=Run_Sweep
#SBATCH --output=Hyper_Sweep.out
#SBATCH --error=Hyper_Sweep_error.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --mem=100Gb

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install torch torchvision
pip3 install tqdm

cd $HOME/GitRepos/DomainBed/

python3 -m domainbed.scripts.sweep delete_incomplete\
       --algorithm IRM VREx SD IGA\
       --dataset TerraIncognita\
       --data_dir $HOME/scratch/data/ \
       --output_dir $HOME/scratch/anneal_experiment/results/hyper_sweep/ \
       --command_launcher multi_gpu \
       --skip_confirmation \
       --n_trials 1

python3 -m domainbed.scripts.sweep launch\
       --algorithm IRM VREx SD IGA\
       --dataset TerraIncognita \
       --data_dir $HOME/scratch/data/ \
       --output_dir $HOME/scratch/anneal_experiment/results/hyper_sweep/ \
       --command_launcher multi_gpu \
       --skip_confirmation \
       --n_trials 1
