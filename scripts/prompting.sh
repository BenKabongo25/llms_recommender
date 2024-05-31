#!/bin/bash

#SBATCH --partition=funky
#SBATCH --job-name=reco_nlp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output=reco_nlp_all_beauty_ratings.out
#SBATCH --error=reco_nlp_all_beauty_ratings.err

python /home/kabongo/reco_nlp/reco_nlp/llms_baseline/prompting.py \
    --base_dir /home/kabongo/reco_nlp/data/process \
    --dataset_name All_Beauty \
    --lang en \
    --verbose \
    --batch_size 16 \
    --evaluate_every 10 \
    --base_data_size 0.25 \
    --max_base_data_samples 2000 \
    --train_size 0.8 \
    --test_size 0.2 \
    --val_size 0.0 \
    --split_method 0 \
    --sampling_method 0 \
    --similarity_function 0 \
    --random_state 42 \
    --n_reviews 4 \
    --n_samples 0 \
    --max_review_length 128 \
    --max_description_length 128 \
    --min_rating 1.0 \
    --max_rating 5.0 \
    --user_id_column user_id \
    --item_id_column item_id \
    --rating_column rating \
    --review_column review \
    --timestamp_flag \
    --timestamp_column timestamp \
    --item_description_flag  \
    --user_description_column description \
    --item_description_column description \
    --source_review_flag \
    --source_rating_flag \
    --user_first_flag \
    --no-target_review_flag \
    --target_rating_flag \
    --max_source_length 1024 \
    --max_target_length 128 \
    --model_name_or_path google/flan-t5-base \
    --tokenizer_name_or_path google/flan-t5-base \