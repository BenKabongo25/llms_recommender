#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=reco_nlp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output=t5_mlp.out
#SBATCH --error=t5_mlp.err

python /home/kabongo/reco_nlp/reco_nlp/llms_baseline/t5_seq_classification.py \
    --model_name_or_path google/flan-t5-base \
    --tokenizer_name_or_path google/flan-t5-base \
    --max_source_length 1024 \
    --max_target_length 128 \
    --base_dir /home/kabongo/reco_nlp/data/process \
    --dataset_name All_Beauty \
    --train_size 0.8 \
    --no-do_classification \
    --n_epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --save_every 10 \
    --base_data_size 0.25 \
    --max_base_data_samples 1000000 \
    --split_method 0 \
    --sampling_method 1 \
    --similarity_function 0 \
    --n_reviews 4 \
    --n_samples 0 \
    --max_review_length 128 \
    --max_description_length 128 \
    --min_rating 1.0 \
    --max_rating 5.0 \
    --timestamp_flag \
    --no-user_description_flag \
    --item_description_flag \
    --no-user_only_flag \
    --source_review_flag \
    --source_rating_flag \
    --target_review_flag \
    --target_rating_flag \
    --user_first_flag \
    