#!/bin.bash             # use bash as command interpreter
#$ -cwd                 # currentWorkingDirectory
#$ -N TSNE              # jobName
#$ -j y                 # merges output and errors
#$ -S /bin/bash         # scripting language
#$ -l h_rt=120:00:00    # jobDuration hh:mm:ss
#$ -q mecc4.q           # queueName
#$ -pe mpi 30           # cpuNumber

conda activate hsi

python tools/reduce_dimensionality.py --reduction_method TSNE --modality absorbance --apply_equalization --input_dir /global-scratch/bulk_pool/vdanilov/projects/hsi_analysis/dataset/HSI --save_dir /global-scratch/bulk_pool/vdanilov/projects/hsi_analysis/dataset
