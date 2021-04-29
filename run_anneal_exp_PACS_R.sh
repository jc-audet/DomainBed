#!/bin/bash
#SBATCH --exclude=cn-b004,cn-c037,cn-b001,cn-a009,cn-c011
#SBATCH --job-name=Anneal_sweep_PACS_R
#SBATCH --output=Anneal_sweep_PACS_R.out
#SBATCH --error=Anneal_sweep_error_PACS_R.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --mem=100Gb

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install torch torchvision
pip3 install tqdm

cd $HOME/GitRepos/DomainBed/

python3 -m domainbed.scripts.anneal_sweep delete_incomplete\
       --algorithm IRM VREx\
       --dataset PACS\
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/anneal_experiment/results/PACS_results_R/\
       --command_launcher multi_gpu\
       --skip_confirmation\
       --steps 300 \
       --n_trials 2 \
       --n_anneal 20

python3 -m domainbed.scripts.anneal_sweep launch\
       --algorithm IRM VREx\
       --dataset PACS\
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/anneal_experiment/results/PACS_results_R/\
       --command_launcher multi_gpu\
       --skip_confirmation\
       --steps 300 \
       --n_trials 2 \
       --n_anneal 20

