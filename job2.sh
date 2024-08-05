#!/bin/bash --login

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # memory per CPU core
#SBATCH -J "GAT 2-class"   # job name
#SBATCH --array=0-1
#SBATCH --gpus=1
##SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE 
mamba activate geo2
python3 -u store_data.py $SLURM_ARRAY_TASK_ID 2 3 10 1000 10 2 "False" "True"
