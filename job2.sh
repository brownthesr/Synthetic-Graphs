#!/bin/bash --login

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # memory per CPU core
#SBATCH -J "GAT 2-class"   # job name
#SBATCH --array=0-200
#SBATCH --gpus=1
##SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE 
mamba activate geo2
# python3 -u generate_data_organized2.py $SLURM_ARRAY_TASK_ID 2 3 10 1000 10 2 "False" "True"
python3 -u generate_data_organized2.py --comp_id $SLURM_ARRAY_TASK_ID --max_comps 200 --mu_max 2 --lambda_min -3 --lambda_max 3 --num_features 10 --num_nodes 1000 --num_classes 2 --average_degree 10 --degree_corrected "False" --hidden_features 16 --lr .01 --runs 1 --epochs 400 --device "cuda" --models GCN