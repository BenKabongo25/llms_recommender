# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Common
# Reviews and ratings evaluation

import evaluate
import numpy as np
from sklearn import metrics
from typing import *


def reviews_evaluation(predictions: List[str], references: List[str], args) -> Dict[str, Any]:
    references_list = [[ref] for ref in references]

    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references_list)
    bleu_results["precision"] = np.mean(bleu_results["precisions"])

    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions, references=references, lang=args.lang
    )
    bertscore_results["precision"] = np.mean(bertscore_results["precision"])
    bertscore_results["recall"] = np.mean(bertscore_results["recall"])
    bertscore_results["f1"] = np.mean(bertscore_results["f1"])

    meteor_metric = evaluate.load("meteor")
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)

    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "n_examples": len(predictions),
        "BERTScore": bertscore_results,
        "BLEU": bleu_results,
        "ROUGE": rouge_results,
        "METEOR": meteor_results,
    }


text_evaluation = reviews_evaluation


def ratings_evaluation(predictions: List[float], references: List[float], args) -> Dict:
    n_non_numerical = 0
    n_examples = len(predictions)
    non_numerical_examples = []    
    numerical_predictions = []
    numerical_references = []

    for i in range(len(predictions)):
        try:
            pr = float(predictions[i])
            rr = float(references[i])
            numerical_predictions.append(pr)
            numerical_references.append(rr)
        except:
            n_non_numerical += 1
            if n_non_numerical < 10:
                non_numerical_examples.append(predictions[i])

    if len(predictions) == 0:
        return {
            "n_examples": n_examples,
            "n_non_numerical": n_non_numerical,
            "non_numerical_examples": non_numerical_examples
        }

    predictions = np.array(numerical_predictions, dtype=float)
    references = np.array(numerical_references, dtype=float)

    mean_rating = np.ceil((args.min_rating + args.max_rating) / 2)
    binary_predictions = np.where(predictions > mean_rating, 1, 0)
    binary_references = np.where(references > mean_rating, 1, 0)

    try:
        auc = metrics.roc_auc_score(binary_references, binary_predictions)
    except:
        auc = None

    return {
        "n_examples": n_examples,
        "n_non_numerical": n_non_numerical,
        "rmse": np.sqrt(metrics.mean_squared_error(references, predictions)),
        "mae": metrics.mean_absolute_error(references, predictions),
        "precision": metrics.accuracy_score(binary_references, binary_predictions),
        "recall": metrics.recall_score(binary_references, binary_predictions),
        "f1": metrics.f1_score(binary_references, binary_predictions),
        "auc": auc,
        "non_numerical_examples": non_numerical_examples
    }


def ratings_aspects_evaluation(
        predictions: Dict[str, List[float]], 
        references: Dict[str, List[float]], 
        args
    ) -> Dict[str, Dict]:
    aspects_scores = {}
    for aspect in predictions:
        aspects_scores[aspect] = ratings_evaluation(predictions[aspect], references[aspect], args)
    return aspects_scores


def evaluate_fn(
    reviews_predictions: List[str], 
    reviews_references: List[str], 
    ratings_predictions: List[float], 
    ratings_references: List[float],
    args
) -> Dict:
    
    reviews_scores = {}
    if len(reviews_predictions) > 0:
        reviews_scores = reviews_evaluation(reviews_predictions, reviews_references, args)
    
    ratings_scores = {}
    if len(ratings_predictions) > 0:
        ratings_scores = ratings_evaluation(ratings_predictions, ratings_references, args)

    return {"reviews": reviews_scores, "ratings": ratings_scores}


if __name__ == "__main__":
    class Args:
        pass
    args = Args()
    args.lang = "en"
    args.min_rating = 1.
    args.max_rating = 5.

    predictions = list(np.random.randint(1, 6, size=(10,))) + ["Bad prediction"]
    references = list(np.random.randint(1, 6, size=(10,))) + [1]
    scores = ratings_evaluation(predictions, references, args)
    print("Ratings evaluation:", scores)

    predictions = [
        "The weather is lovely today.", 
        "I enjoyed the movie very much.", 
        "The team played exceptionally well."
    ]
    references = [
        "The weather is beautiful today.", 
        "I really liked the movie.", 
        "The team performed outstandingly."
    ]

    scores = reviews_evaluation(predictions, references, args)
    print("Good pred.:", scores)

    predictions = [
        "I hate the rain, it's terrible.", 
        "The movie was boring, I didn't like it at all.", 
        "The team lost miserably, they played very badly."
    ]
    references = [
        "I love rainy days, they're cozy.", 
        "The movie was captivating, I thoroughly enjoyed it.", 
        "The team won by a landslide, their performance was exceptional."
    ]

    scores = reviews_evaluation(predictions, references, args)
    print("Bad pred.:", scores)
