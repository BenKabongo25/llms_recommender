#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=p5_small
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6000
#SBATCH --output=p5_small_beauty.out
#SBATCH --error=p5_small_beauty.err

python /home/kabongo/reco_nlp/reco_nlp/llms_baseline/t5_seq_classification.py \
    --model_name_or_path google/flan-t5-small \
    --tokenizer_name_or_path google/flan-t5-small \
    --max_source_length 1024 \
    --max_target_length 8 \
    --base_dir /home/kabongo/reco_nlp/data/process \
    --dataset_name All_Beauty \
    --train_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/train/sampled_data.csv \
    --test_dataset_path /home/kabongo/reco_nlp/data/process/All_Beauty/samples/dir/test/sampled_data.csv \
    --exp_name p5_small_rating_0 \
    --prompt_type 0 \
    --train_size 0.8 \
    --n_epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --save_every 10 \
    --max_review_length 128 \
    --max_description_length 128 \
    --min_rating 1.0 \
    --max_rating 5.0 \
    --timestamp_flag \
    --no-user_description_flag \
    --item_description_flag \
    