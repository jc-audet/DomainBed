#!/bin/bash
#SBATCH --job-name=Anneal_sweep_SD
#SBATCH --output=Anneal_sweep_SD.out
#SBATCH --error=Anneal_sweep_error_SD.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mem=100Gb
#SBATCH --exclude=cn-b004,cn-c037,cn-b001,cn-a009,cn-c011

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install torch torchvision
pip3 install tqdm

cd $HOME/GitRepos/DomainBed/

python3 -m domainbed.scripts.anneal_sweep launch\
       --algorithm SD\
       --dataset ColoredMNIST CFMNIST CSMNIST ACMNIST\
       --data_dir $HOME/scratch/data/MNIST/\
       --output_dir $HOME/scratch/anneal_experiment/results/SD/\
       --command_launcher multi_gpu\
       --skip_confirmation\
       --steps 2000\
       --n_trials 3\
       --n_anneal 20

