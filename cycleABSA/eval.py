# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Eval

import evaluate
import numpy as np
from typing import *

from annotations_text import AnnotationsTextFormerBase
from enums import TaskType


def text_evaluation(predictions: List[str], references: List[str], args: Any) -> Dict[str, Any]:
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
    del bertscore_results["hashcode"]

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
        "n_pred": N_pred,
        "n_true": N_true,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    

def get_evaluation_scores(
    predictions: List[str], 
    references: Union[List[str], List[List[Tuple[str]]]], 
    annotations: List[List[Tuple[str]]],
    annotations_text_former: AnnotationsTextFormerBase, 
    args: Any,
    task_type: Optional[TaskType]=None
) -> Dict[str, Dict]:
    if task_type is None:
        task_type = args.task_type
        
    if task_type is TaskType.A2T:
        scores = text_evaluation(predictions, references, args)
    else:
        predictions = [
            annotations_text_former.multiple_text_to_annotations(pred)
            for pred in predictions
        ]
        scores = aspect_evaluation(predictions, annotations, args)
    return scores


if __name__ == "__main__":
    def test():
        pred = [[('atmosphere', 'ambience general', 'positive', 'relaxed')]]
        true = [[
            ('atmosphere', 'ambience general', 'positive', 'relaxed'),
            ('atmosphere&', 'ambience general', 'positive', 'relaxed')
        ]]
        scores = aspect_evaluation(pred, true, None)
        print(scores)

    test()