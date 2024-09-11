# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Absa: Prompts

from typing import Any, Tuple

from enums import PrompterType, TaskType


class PrompterBase(object):

    def __init__(self, args: Any):
        self.args = args

    def get_prompt(
        self, 
        task_type: TaskType,
        text: str=None, 
        annotations: Tuple[str]=None, 
        polarities: Tuple[str]=None
    ) -> str:
        raise NotImplementedError

    def get_text(self, task_type: TaskType, prompt: str) -> str:
        raise NotImplementedError

    @classmethod
    def get_prompter(self, args: Any):
        if args.prompter_type is PrompterType.P1:
            return Prompter1(args)
        else:
            return Prompter2(args)


class Prompter1(PrompterBase):

    def get_prompt(
        self, 
        task_type: TaskType=None,
        text: str=None, 
        annotations: Tuple[str]=None, 
        polarities: Tuple[str]=None
    ) -> str:
        if task_type is None:
            task_type = self.args.task_type
        if task_type is TaskType.T2A:
            return "translate from text to absa tuples: {text}".format(text=text)
        else:
            return "translate from absa tuples to text: {text}".format(text=text)

    def get_text(self, task_type: TaskType, prompt: str) -> str:
        if task_type is TaskType.T2A:
            prefix = "translate from text to absa tuples: "
        else:
            prefix = "translate from absa tuples to text: "
        text = prompt[len(prefix):]
        return text


class Prompter2(PrompterBase):
    PROMPTS = {
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

            "acop": "Extract quadruples of aspect terms, aspect categories, opinion terms, and sentiment polarities {polarities} from the following text: {text}"
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
        self, 
        task_type: TaskType,
        text: str=None, 
        annotations: Tuple[str]=None, 
        polarities: Tuple[str]=None
    ) -> str:
        if task_type is None:
            task_type = self.args.task_type
        prompt = Prompter2.PROMPTS[task_type.value][self.args.absa_tuple.value]
        prompt = prompt.format(
            text=text,
            annotations=annotations,
            polarities=polarities
        )
        return prompt


if __name__ == "__main__":
    from enums import AbsaTupleType
    class Args:
        pass

    def test():
        args = Args()
        args.task_type = TaskType.T2A
        args.absa_tuple = AbsaTupleType.ACOP
        text = "The battery life of this laptop is very good."
        prompter = Prompter2(args)
        prompt = prompter.get_prompt(text=text)
        print(prompt)

        args.task_type = TaskType.A2T
        args.absa_tuple = AbsaTupleType.ACOP
        annotations = ("battery life", "quality", "good", "positive")
        text = str(annotations)
        prompt = prompter.get_prompt(text=text, annotations=annotations)
        print(prompt)

    test()