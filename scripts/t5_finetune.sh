#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=p5_small
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output=t5_finetune_beauty.out
#SBATCH --error=t5_finetune_beauty.err

python /home/kabongo/reco_nlp/reco_nlp/llms_baseline/t5_finetuning.py \
    --model_name_or_path google/flan-t5-small \
    --tokenizer_name_or_path google/flan-t5-small \
    --max_source_length 1024 \
    --max_target_length 8 \
    --base_dir /home/kabongo/reco_nlp/data/process \
    --dataset_name All_Beauty \
    --train_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/train/sampled_data.csv \
    --test_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/test/sampled_data.csv \
    --exp_name t5_small_rating \
    --train_size 0.8 \
    --n_epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --save_every 10 \
    --max_review_length 128 \
    --max_description_length 128 \
    --min_rating 1.0 \
    --max_rating 5.0 \
    --no-user_description_flag \
    --item_description_flag \
    --base_data_size 0.25 \
    --max_base_data_samples 10000 \
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
    