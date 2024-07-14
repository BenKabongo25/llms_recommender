# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Prompts

from typing import Any, Tuple


prompts = {
    "T2A": {

        "a": "Extract all aspect terms from the following text: {text}",
        "c": "Extract all aspect categories from the following text: {text}",
        "o": "Extract all opinion terms from the following text: {text}",
        "p": "Determine the sentiment polarity {polarities} of the following text: {text}",

        "ac": "Extract pairs of aspect terms and their corresponding aspect categories from the following text: {text}",
        "ao": "Extract pairs of aspect terms and their corresponding opinion terms from the following text: {text}",
        "ap": "Extract pairs of aspect terms and their corresponding sentiment polarities {polarities} from the following text: {text}",
        "co": "Extract pairs of aspect categories and their corresponding opinion terms from the following text: {text}",
        "cp": "Extract pairs of aspect categories and their corresponding sentiment polarities {polarities} from the following text: {text}",
        "op": "Extract pairs of opinion terms and their corresponding sentiment polarities {polarities} from the following text: {text}",

        "aco": "Extract triples of aspect terms, aspect categories, and opinion terms from the following text: {text}",
        "acp": "Extract triples of aspect terms, aspect categories, and sentiment polarities {polarities} from the following text: {text}",
        "aop": "Extract triples of aspect terms, opinion terms, and sentiment polarities {polarities} from the following text: {text}",
        "cop": "Extract triples of aspect categories, opinion terms, and sentiment polarities {polarities} from the following text: {text}",

        "acop": "Extract quadruples of aspect terms, aspect categories, opinion terms, and sentiment polarities {polarities} from the following text: {text}",
    },

    "A2T": {
        "a": "Generate a text that includes the following aspect terms: {annotations}",
        "c": "Generate a text that includes the following aspect categories: {annotations}",
        "o": "Generate a text that includes the following opinion terms: {annotations}",
        "p": "Generate a text that includes the following sentiment polarities: {annotations}",

        "ac": "Generate a text that includes the following pairs of aspect terms and their corresponding aspect categories: {annotations}",
        "ao": "Generate a text that includes the following pairs of aspect terms and their corresponding opinion terms: {annotations}",
        "ap": "Generate a text that includes the following pairs of aspect terms and their corresponding sentiment polarities: {annotations}",
        "co": "Generate a text that includes the following pairs of aspect categories and their corresponding opinion terms: {annotations}",
        "cp": "Generate a text that includes the following pairs of aspect categories and their corresponding sentiment polarities: {annotations}",
        "op": "Generate a text that includes the following pairs of opinion terms and their corresponding sentiment polarities: {annotations}",

        "aco": "Generate a text that includes the following triples of aspect terms, aspect categories, and opinion terms: {annotations}",
        "acp": "Generate a text that includes the following triples of aspect terms, aspect categories, and sentiment polarities: {annotations}",
        "aop": "Generate a text that includes the following triples of aspect terms, opinion terms, and sentiment polarities: {annotations}",
        "cop": "Generate a text that includes the following triples of aspect categories, opinion terms, and sentiment polarities: {annotations}",

        "acop": "Generate a text that includes the following quadruples of aspect terms, aspect categories, opinion terms, and sentiment polarities: {annotations}",
    }
}


def get_prompt(
    text: str=None,
    annotations: Tuple[str]=None,
    polarities: Tuple[str]=None,
    args: Any=None
) -> str:
    prompt = prompts[args.task_name][args.absa_tuple]
    prompt = prompt.format(
        text=text,
        annotations="(" + ", ".join(annotations) + ")" if annotations else "",
        polarities="(" + ", ".join(polarities) + ")" if polarities else ""
    )
    return prompt


if __name__ == "__main__":
    class Args:
        pass

    def test():
        args = Args()
        args.task_name = "T2A"
        args.absa_tuple = "acop"
        text = "The battery life of this laptop is very good."
        prompt = get_prompt(text=text, args=args)
        print(prompt)

        args.task_name = "A2T"
        args.absa_tuple = "acop"
        annotations = ("battery life", "quality", "good", "positive")
        prompt = get_prompt(annotations=annotations, args=args)
        print(prompt)

    test()