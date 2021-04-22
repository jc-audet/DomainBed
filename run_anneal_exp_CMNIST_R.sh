#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --job-name=Anneal_sweep_CMNIST_with_reset
#SBATCH --output=Anneal_sweep_CMNIST_with_reset.out
#SBATCH --error=Anneal_sweep_error_CMNIST_with_reset.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --mem=100Gb

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index torch torchvision
pip3 install --no-index tqdm

cd $HOME/GitRepos/DomainBed/

# Copy data to compute node
cp -r $SCRATCH/data/MNIST $SLURM_TMPDIR
	
python3 -m domainbed.scripts.train\
       --data_dir $SLURM_TMPDIR/MNIST\
       --output_dir $SLURM_TMPDIR/misc/ColoredMNIST_results_ERM/1/\
       --algorithm ERM \
       --dataset ColoredMNIST \
       --steps 2000 \
       --seed 1 \
       --trial_seed 1
       
python3 -m domainbed.scripts.train\
       --data_dir $SLURM_TMPDIR/MNIST\
       --output_dir $SLURM_TMPDIR/misc/ColoredMNIST_results_ERM/2/\
       --algorithm ERM \
       --dataset ColoredMNIST \
       --steps 2000 \
       --seed 2 \
       --trial_seed 2
       
python3 -m domainbed.scripts.train\
       --data_dir $SLURM_TMPDIR/MNIST\
       --output_dir $SLURM_TMPDIR/misc/ColoredMNIST_results_ERM/3/\
       --algorithm ERM \
       --dataset ColoredMNIST \
       --steps 2000 \
       --seed 3 \
       --trial_seed 3

python3 -m domainbed.scripts.anneal_sweep launch\
       --algorithm SD ANDMask IRM IGA VREx\
       --dataset ColoredMNIST\
       --data_dir $SLURM_TMPDIR/MNIST/\
       --output_dir $SLURM_TMPDIR/misc/ColoredMNIST_R_results/\
       --command_launcher multi_gpu\
       --skip_confirmation\
       --steps 2000\
       --n_trials 3\
       --n_anneal 20

cp -r $SLURM_TMPDIR/misc $SCRATCH/anneal_experiment/


