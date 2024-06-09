#!/bin/bash

#SBATCH --partition=funky
#SBATCH --job-name=reco_nlp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output= amazon_process.out
#SBATCH --error= amazon_process.err

python /home/kabongo/reco_nlp/reco_nlp/common/data/sample_data.py \
    --base_dir /home/kabongo/reco_nlp/data/process \
    --dataset_name Electronics \