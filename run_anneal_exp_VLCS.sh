#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --job-name=Anneal_sweep_VLCS
#SBATCH --output=Anneal_sweep_VLCS.out
#SBATCH --error=Anneal_sweep_error_VLCS.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mem=100Gb

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index torch torchvision
pip3 install --no-index tqdm

# Copy data to compute node
cp -r $SCRATCH/data/VLCS.tar.gz $SLURM_TMPDIR
python3 -m domainbed.scripts.download \
       --data_dir=$SLURM_TMPDIR\
       --dataset=VLCS

cd $HOME/GitRepos/DomainBed/

python3 -m domainbed.scripts.anneal_sweep delete_incomplete\
       --algorithm ERM SD ANDMask IRM IGA VREx\
       --dataset VLCS\
       --data_dir $SLURM_TMPDIR/\
       --output_dir $SLURM_TMPDIR/misc/VLCS_results/\
       --command_launcher multi_gpu\
       --steps 2000\
	--skip_confirmation\
	--n_trials 3\
	--n_anneal 20

python3 -m domainbed.scripts.anneal_sweep launch\
       --algorithm ERM SD ANDMask IRM IGA VREx\
       --dataset VLCS\
       --data_dir $SLURM_TMPDIR/\
       --output_dir $SLURM_TMPDIR/misc/VLCS_results/\
       --command_launcher multi_gpu\
       --skip_confirmation\
       --steps 2000\
       --n_trials 3\
       --n_anneal 20

cp -r $SLURM_TMPDIR/misc $SCRATCH/anneal_experiment/


