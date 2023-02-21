#!/bin.bash             # use bash as command interpreter
#$ -cwd                 # currentWorkingDirectory
#$ -N TSNE              # jobName
#$ -j y                 # merges output and errors
#$ -S /bin/bash         # scripting language
#$ -l h_rt=120:00:00    # jobDuration hh:mm:ss
#$ -q mecc4.q           # queueName
#$ -pe mpi 30           # cpuNumber

conda activate hsi

python src/data/reduce_dimensionality.py reduction_method=tsne modality=abs apply_equalization=true input_dir=/global-scratch/bulk_pool/vdanilov/projects/hsi_analysis/dataset/raw save_dir=/global-scratch/bulk_pool/vdanilov/projects/hsi_analysis/data/sly_input
