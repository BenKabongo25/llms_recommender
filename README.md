# Are Natural Language User-Item Interactions All You Need?  
### Evaluating Large Language Models on the Prediction of User Rating and Review

## Introduction
This repository contains the implementation for our research on evaluating the performance of **Large Language Models (LLMs)** in recommendation tasks. The key focus of our work is to explore how LLMs can leverage textual user-item interactions, such as reviews and descriptions, for both **rating prediction** and **review prediction**. Traditional recommender systems, like **Matrix Factorization (MF)**, often rely solely on numerical ratings. In contrast, LLMs provide the opportunity to integrate richer textual data to enhance recommendations.

This project provides the code for two tasks:
- **Rating prediction**: Predicting the numerical rating a user would give to an item.
- **Review prediction**: Predicting the textual review a user would write for an item.

We compare various LLM architectures and sizes across different datasets using **prompting** (zero-shot and few-shot) and **fine-tuning** approaches.

## Repository Structure

```
llms_recommender/
│
├── data/                   # Data processing scripts for different datasets
│   ├── amazon_review_process.py
│   ├── beer_process.py
│   ├── sample_data.py
│   └── tripadvisor.py
│
├── utils/                  # Utility functions for evaluation, data preprocessing
│   ├── evaluation.py       # Functions for calculating metrics such as RMSE, MAE, Precision, Recall, etc.
│   ├── functions.py        # General helper functions
│   ├── preprocess_text.py  # Functions for text preprocessing
│   └── vocabulary.py
│
├── datasets/               # Folder for storing dataset files and URLs
│   └── urls.txt            # Links to download datasets used in the paper
│
├── llms/                   # Code related to LLM prompting, fine-tuning, and sampling
│   ├── bart_conditional_generation.py
│   ├── data.py
│   ├── p5_finetuning.py
│   ├── prompters.py        # Functions for generating prompts for LLMs
│   ├── prompting.py        # Main script for prompt-based evaluations
│   ├── t5_conditional_generation.py
│   ├── t5_seq_classification.py
│   └── utils.py
│
├── reco_baselines/         # Implementation of classical recommender system methods
│   ├── mf.py               # Matrix Factorization implementation
│   ├── mlp.py              # Multi-Layer Perceptron implementation
│   └── stats.py            # Baseline methods (e.g., global and item average rating)
│
└── README.md               # Project overview (this file)
```

## Datasets
We conduct experiments on several datasets from different domains, each consisting of reviews, ratings, and optionally descriptions of users and items. The datasets include:

- **Amazon Product Reviews** (Beauty, CDs, etc.)
- **Beer Multi-Aspects Reviews** (RateBeer)
- **Hotel Multi-Aspects Reviews** (TripAdvisor)

To use these datasets, download them using the links provided in the `datasets/urls.txt` file.

## Methodology
Our approach focuses on two main tasks:
1. **Rating Prediction**: Predict the numerical rating for a user-item pair. LLMs are either prompted (zero-shot, few-shot) or fine-tuned for the task.
2. **Review Prediction**: Generate a review text for a user-item pair. Evaluation metrics include BLEU, ROUGE, METEOR, and BERTScore.

### Prompting and Fine-Tuning
We test both **prompt-based** and **fine-tuned** LLM models. The LLMs used in our experiments include:
- **Flan-T5** (small to XXL)
- **P5** (T5-based pre-trained for recommendation tasks)

Baselines include **Matrix Factorization (MF)** and **Multi-Layer Perceptron (MLP)** models. We compare these classical methods against our LLM-based methods.

## Evaluation Metrics
### For Rating Prediction:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Precision**, **Recall**, **F1 Score** (based on thresholded ratings)

### For Review Prediction:
- **BLEU**
- **ROUGE**
- **METEOR**
- **BERTScore**
     ```
