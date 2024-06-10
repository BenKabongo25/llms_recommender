#!/bin/bash

#SBATCH --partition=funky
#SBATCH --job-name=reco_nlp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output=mf.out
#SBATCH --error=mf.err

algos=("svd" "svd++" "nmf")
for algo in "${algos[@]}"
do
    python /home/kabongo/reco_nlp/reco_nlp/reco_baselines/mf.py \
        --base_dir /home/kabongo/reco_nlp/data/process \
        --dataset_name All_Beauty \
        --train_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/train/sampled_data.csv \
        --test_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/test/sampled_data.csv \
        --algo $algo \
        --n_factors 100 \
        --n_epochs 30 \
        --biased \
        --lr_all 0.005 \
        --reg_all 0.02 \
        --min_rating 1.0 \
        --max_rating 5.0 \
        --train_size 0.8 \
        --user_id_column user_id \
        --item_id_column item_id \
        --rating_column rating \
        --random_state 42 \
        --verbose
done