#!/bin/bash
#SBATCH --job-name=Anneal_sweep_VLCS_R
#SBATCH --output=Anneal_sweep_VLCS_R.out
#SBATCH --error=Anneal_sweep_error_VLCS_R.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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

python3 -m domainbed.scripts.anneal_sweep delete_incomplete\
       --algorithm IRM VREx \
       --dataset VLCS \
       --data_dir $HOME/scratch/data/ \
       --output_dir $HOME/scratch/anneal_experiment/results/VLCS_results_R/ \
       --command_launcher multi_gpu \
       --skip_confirmation \
       --steps 300 \
       --n_trials 2 \
       --n_anneal 20

python3 -m domainbed.scripts.anneal_sweep launch\
       --algorithm IRM VREx \
       --dataset VLCS \
       --data_dir $HOME/scratch/data/ \
       --output_dir $HOME/scratch/anneal_experiment/results/VLCS_results_R/ \
       --command_launcher multi_gpu \
       --skip_confirmation \
       --steps 300 \
       --n_trials 2 \
       --n_anneal 20
