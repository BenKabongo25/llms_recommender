# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Eval

import evaluate
import numpy as np
from typing import *

from utils import get_annotations


def text_evaluation(
    predictions: List[str], 
    references: List[str], 
    args: Any
) -> Dict[str, Any]:
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


def aspect_evaluation(
    predictions: List[List[Tuple[str]]],
    references: List[List[Tuple[str]]],
    args: Any
) -> Dict[str, Union[int, float]]:
    TP = 0
    N_pred = 0
    N_true = 0

    for i in range(len(predictions)):
        pred = set(predictions[i])
        true = set(references[i])

        TP += len(pred.intersection(true))
        N_pred += len(pred)
        N_true += len(true)

    precision = TP / N_pred if N_pred > 0 else 0
    recall = TP / N_true if N_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "n_examples": len(predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    

def get_evaluation_scores(
    predictions: List[str], 
    references: List[str], 
    annotations: List[List[Tuple[str]]],
    args: Any
) -> Dict[str, Dict]:
    if args.task_name == "A2T":
        scores = text_evaluation(predictions, references, args)
    else: # args.task_name == "T2A"
        predictions = [get_annotations(pred, args) for pred in predictions]
        scores = aspect_evaluation(predictions, annotations, args)
    return scores
