#!/bin/bash
#SBATCH --job-name=Anneal_sweep_ACMNIST_with_reset
#SBATCH --output=Anneal_sweep_ACMNIST_with_reset.out
#SBATCH --error=Anneal_sweep_error_ACMNIST_with_reset.out
#SBATCH -–partition=long
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

python3 -m domainbed.scripts.train\
       --data_dir $SLURM_TMPDIR/MNIST\
       --output_dir $HOME/scratch/anneal_experiment/results/ACMNIST_ERM/1/\
       --algorithm ERM \
       --dataset ACMNIST \
       --steps 2000 \
       --seed 1 \
       --trial_seed 1
       
python3 -m domainbed.scripts.train\
       --data_dir $SLURM_TMPDIR/MNIST\
       --output_dir $HOME/scratch/anneal_experiment/results/ACMNIST_ERM/2/\
       --algorithm ERM \
       --dataset ACMNIST \
       --steps 2000 \
       --seed 2 \
       --trial_seed 2
       
python3 -m domainbed.scripts.train\
       --data_dir $SLURM_TMPDIR/MNIST\
       --output_dir $HOME/scratch/anneal_experiment/results/ACMNIST_ERM/3/\
       --algorithm ERM \
       --dataset ACMNIST \
       --steps 2000 \
       --seed 3 \
       --trial_seed 3

python3 -m domainbed.scripts.anneal_sweep launch\
       --algorithm SD ANDMask IRM IGA VREx\
       --dataset ACMNIST\
       --data_dir $SLURM_TMPDIR/MNIST/\
       --output_dir $HOME/scratch/anneal_experiment/results/ACMNIST_R/\
       --command_launcher multi_gpu\
       --skip_confirmation\
       --steps 2000\
       --n_trials 3\
       --n_anneal 20

