#!/bin/bash

#SBATCH --partition=funky
#SBATCH --job-name=reco_nlp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output=mlp.out
#SBATCH --error=mlp.err

python /home/kabongo/reco_nlp/reco_nlp/reco_baselines/mlp.py \
    --base_dir /home/kabongo/reco_nlp/data/process \
    --dataset_name All_Beauty \
    --train_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/train/sampled_data.csv \
    --test_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/test/sampled_data.csv \
    --embedding_dim 32 \
    --padding_idx 0 \
    --no-do_classification \
    --min_rating 1.0 \
    --max_rating 5.0 \
    --n_epochs 30 \
    --batch_size 64 \
    --train_size 0.8 \
    --lr 0.001 \
    --user_id_column user_id \
    --item_id_column item_id \
    --rating_column rating \
    --user_vocab_id_column user_vocab_id \
    --item_vocab_id_column item_vocab_id \
    --random_state 42 \
    --verbose \
    --verbose_every 10 \