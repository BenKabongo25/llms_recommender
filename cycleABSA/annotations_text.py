# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Annotations to text and text to annotations

from typing import *

from enums import AbsaTupleType, AnnotationsTextFormerType, TaskType
from utils import AbsaData


class AnnotationsTextFormerBase(object):

    def __init__(self, absa_data: AbsaData, args: Any):
        self.absa_data = absa_data
        self.args = args

    def annotations_to_text(self, annotations: Tuple[str]) -> str:
        raise NotImplementedError

    def text_to_annotations(self, text: str) -> Tuple[str]:
        raise NotImplementedError

    def multiple_annotations_to_text(self, annotations: List[Tuple[str]]) -> str:
        raise NotImplementedError

    def multiple_text_to_annotations(self, text: str) -> List[Tuple[str]]:
        raise NotImplementedError

    @classmethod
    def get_annotations_text_former(cls, args, absa_data):
        if args.annotations_text_type is AnnotationsTextFormerType.GAS_ANNOTATION_STYLE:
            annotations_text_former = GASAnnotationStyleAnnotationsTextFormer(absa_data, args)
        elif args.annotations_text_type is AnnotationsTextFormerType.GAS_EXTRACTION_STYLE:
            annotations_text_former = GASExtractionStyleAnnotationsTextFormer(absa_data, args)
        else:
            annotations_text_former = ParaphraseAnnotationsTextFormer(absa_data, args)
        return annotations_text_former


class GASExtractionStyleAnnotationsTextFormer(AnnotationsTextFormerBase):
    """
    Towards Generative Aspect-Based Sentiment Analysis paper
    https://aclanthology.org/2021.acl-short.64/
    """

    def __init__(self, absa_data: AbsaData, args: Any):
        super().__init__(absa_data, args)

        self.annotations_separator = ";"
        self.annotations_begin = "("
        self.annotations_end = ")"
        self.annotations_element_separator = ","

    def annotations_to_text(self, annotations: Tuple[str]) -> str:
        text = self.annotations_begin
        text += (self.annotations_element_separator + " ").join(annotations)
        text += self.annotations_end
        return text

    def text_to_annotations(self, text: str) -> Tuple[str]:
        text = text.strip()
        if not (text.startswith("(") and text.endswith(")")):
            return ()
        text = text[1:-1]
        annotations = tuple([t.strip() for t in text.split(",")])
        return annotations

    def multiple_annotations_to_text(self, annotations: List[Tuple[str]]) -> str:
        annotations_texts = [self.annotations_to_text(ann) for ann in annotations]
        text = (self.annotations_separator + " ").join(annotations_texts)
        text += self.annotations_separator
        return text

    def multiple_text_to_annotations(self, text: str) -> List[Tuple[str]]:
        annotations = []
        for text in text.split(self.annotations_separator):
            ann = self.text_to_annotations(text)
            if ann != ():
                annotations.append(ann)
        return annotations


class GASAnnotationStyleAnnotationsTextFormer(AnnotationsTextFormerBase):
    """
    Towards Generative Aspect-Based Sentiment Analysis paper
    https://aclanthology.org/2021.acl-short.64/
    """

    def __init__(self, absa_data: AbsaData, args: Any):
        super().__init__(absa_data, args)

    def annotations_to_text(self, annotations: Tuple[str]) -> str:
        raise NotImplementedError

    def text_to_annotations(self, text: str) -> Tuple[str]:
        raise NotImplementedError

    def multiple_annotations_to_text(self, annotations: List[Tuple[str]]) -> str:
        raise NotImplementedError

    def multiple_text_to_annotations(self, text: str) -> List[Tuple[str]]:
        raise NotImplementedError


class ParaphraseAnnotationsTextFormer(AnnotationsTextFormerBase):
    """
    Aspect Sentiment Quad Prediction as Paraphrase Generation
    https://aclanthology.org/2021.emnlp-main.726/
    """

    def __init__(self, absa_data: AbsaData, args: Any):
        super().__init__(absa_data, args)
        self.annotations_separator = ";"

    def aspect_term_annotation_to_paraphrase(self, a: str) -> str:
        return "it" if a.lower() == "null" else a

    def aspect_term_paraphrase_to_annotation(self, a: str) -> str:
        return "null" if a.lower() == "it" else a

    def polarity_annotation_to_paraphrase(self, p: str) -> str:
        return self.absa_data.sentiment_paraphrases_kv.get(p, "")

    def polarity_paraphrase_to_annotation(self, p: str) -> str:
        return self.absa_data.sentiment_paraphrases_vk.get(p, "")

    def quad_elements_to_text(self, pa: str, pc: str, po: str, pp: str) -> str:
        return f"{pc} is {pp} because {pa} is {po}"

    def quad_text_to_elements(self, text: str) -> tuple[str]:
        splits = text.split(" because ")
        if len(splits) < 2:
            return (None, None, None, None)
        pcp, pao = splits[0].strip(), splits[1].strip()

        pcp_splits = pcp.strip().split(" is ")
        if len(pcp_splits) < 2:
            pc, pp = None, None
        else:
            pc, pp = pcp_splits[0].strip(), pcp_splits[1].strip()

        pao_splits = pao.strip().split(" is ")
        if len(pao_splits) < 2:
            pa, po = None, None
        else:
            pa, po = pao_splits[0].strip(), pao_splits[1].strip()
        
        return (pa, pc, po, pp)
        
    def pair_elements_to_text(self, ao: str, cp: str) -> str:
        return f"{ao} is equivalent to {cp}"

    def pair_text_to_elements(self, text: str) -> str:
        splits = text.split(" is equivalent to ")
        if len(splits) < 2:
            return (None, None)
        else:
            ao, cp = splits[0].strip(), splits[1].strip()
            return (ao, cp)

    def annotations_to_text(self, annotations: Tuple[str]) -> str:
        if self.args.absa_tuple in (
            AbsaTupleType.A, AbsaTupleType.C, AbsaTupleType.O, AbsaTupleType.P
        ):
            if self.args.absa_tuple is AbsaTupleType.A:
                (a,) = annotations
                text = self.aspect_term_annotation_to_paraphrase(a)

            elif self.args.absa_tuple is AbsaTupleType.P:
                (p,) = annotations
                text = self.polarity_annotation_to_paraphrase(p)

            else: # C, O
                (co,) = annotations
                text = co

        elif self.args.absa_tuple is AbsaTupleType.AC:
            (a, c) = annotations
            pa = self.aspect_term_annotation_to_paraphrase(a)
            text = self.pair_elements_to_text(pa, c)

        elif self.args.absa_tuple is AbsaTupleType.OP:
            (o, p) = annotations
            pp = self.polarity_annotation_to_paraphrase(p)
            text = self.pair_elements_to_text(o, pp)

        else:
            pa, pc, po, pp = "", "", "", ""

            if self.args.absa_tuple is AbsaTupleType.AO:
                (a, o) = annotations
                pa = self.aspect_term_annotation_to_paraphrase(a)
                pc = pa
                po = o
                pp = o

            elif self.args.absa_tuple is AbsaTupleType.AP:
                (a, p) = annotations
                pa = self.aspect_term_annotation_to_paraphrase(a)
                pp = self.polarity_annotation_to_paraphrase(p)
                pc = pa
                po = pp

            elif self.args.absa_tuple is AbsaTupleType.CO:
                (c, o) = annotations
                pa = c
                pc = c
                po = o
                pp = o

            elif self.args.absa_tuple is AbsaTupleType.CP:
                (c, p) = annotations
                pp = self.polarity_annotation_to_paraphrase(p)
                pa = c
                pc = c
                po = pp

            if self.args.absa_tuple is AbsaTupleType.ACO:
                (a, c, o) = annotations
                pa = self.aspect_term_annotation_to_paraphrase(a)
                pc = c
                po = o
                pp = o

            elif self.args.absa_tuple is AbsaTupleType.ACP:
                (a, c, p) = annotations
                pa = self.aspect_term_annotation_to_paraphrase(a)
                pp = self.polarity_annotation_to_paraphrase(p)
                pc = c
                po = pp

            elif self.args.absa_tuple is AbsaTupleType.AOP:
                (a, o, p) = annotations
                pa = self.aspect_term_annotation_to_paraphrase(a)
                pp = self.polarity_annotation_to_paraphrase(p)
                pc = pa
                po = o

            elif self.args.absa_tuple is AbsaTupleType.COP:
                (c, o, p) = annotations
                pp = self.polarity_annotation_to_paraphrase(p)
                pa = c
                pc = c
                po = o
            
            elif self.args.absa_tuple is AbsaTupleType.ACOP:
                (a, c, o, p) = annotations
                pa = self.aspect_term_annotation_to_paraphrase(a)
                pp = self.polarity_annotation_to_paraphrase(p)
                pc = c
                po = o
                
            text = self.quad_elements_to_text(pa, pc, po, pp)

        return text
            
    def text_to_annotations(self, text: str) -> Tuple[str]:
        text = text.strip()

        if self.args.absa_tuple in (
            AbsaTupleType.A, AbsaTupleType.C, AbsaTupleType.O, AbsaTupleType.P
        ):
            if self.args.absa_tuple is AbsaTupleType.A:
                a = text
                a = self.aspect_term_paraphrase_to_annotation(a)
                if a is not None:
                    return (a,)

            elif self.args.absa_tuple is AbsaTupleType.P:
                p = text
                p = self.polarity_paraphrase_to_annotation(p)
                if p is not None:
                    return (p,)

            else: # C, O
                co = text
                if co is not None:
                    return (co,)
                
        elif self.args.absa_tuple is AbsaTupleType.AC:
            (a, c) = self.pair_text_to_elements(text)
            if a is not None:
                a = self.aspect_term_paraphrase_to_annotation(a)
            if a is not None or c is not None:
                return (a, c)

        elif self.args.absa_tuple is AbsaTupleType.OP:
            (o, p) = self.pair_text_to_elements(text)
            if p is not None:
                p = self.polarity_paraphrase_to_annotation(p)
            if o is not None or p is not None: 
                return (o, p)

        else:
            a, c, o, p = self.quad_text_to_elements(text)
            
            if (a is None) and (c is None) and (o is None) and (p is None):
                return ()

            elif self.args.absa_tuple is AbsaTupleType.AO:
                if a is not None:
                    a = self.aspect_term_paraphrase_to_annotation(a)
                return (a, o)

            elif self.args.absa_tuple is AbsaTupleType.AP:
                if a is not None:
                    a = self.aspect_term_paraphrase_to_annotation(a)
                if p is not None:
                    p = self.polarity_paraphrase_to_annotation(p)
                return (a, p)

            elif self.args.absa_tuple is AbsaTupleType.CO:
                return (c, o)

            elif self.args.absa_tuple is AbsaTupleType.CP:
                if p is not None:
                    p = self.polarity_paraphrase_to_annotation(p)
                return (c, p)

            if self.args.absa_tuple is AbsaTupleType.ACO:
                if a is not None:
                    a = self.aspect_term_paraphrase_to_annotation(a)
                return (a, c, o)

            elif self.args.absa_tuple is AbsaTupleType.ACP:
                if a is not None:
                    a = self.aspect_term_paraphrase_to_annotation(a)
                if p is not None:
                    p = self.polarity_paraphrase_to_annotation(p)
                return (a, c, p)

            elif self.args.absa_tuple is AbsaTupleType.AOP:
                if a is not None:
                    a = self.aspect_term_paraphrase_to_annotation(a)
                if p is not None:
                    p = self.polarity_paraphrase_to_annotation(p)
                return (a, o, p)
                
            elif self.args.absa_tuple is AbsaTupleType.COP:
                if p is not None:
                    p = self.polarity_paraphrase_to_annotation(p)
                return (c, o, p)
            
            elif self.args.absa_tuple is AbsaTupleType.ACOP:
                if a is not None:
                    a = self.aspect_term_paraphrase_to_annotation(a)
                if p is not None:
                    p = self.polarity_paraphrase_to_annotation(p)
                return (a, c, o, p)
                
        return ()

    def multiple_annotations_to_text(self, annotations: List[Tuple[str]]) -> str:
        annotations_texts = [self.annotations_to_text(ann) for ann in annotations]
        text = (self.annotations_separator + " ").join(annotations_texts)
        text += self.annotations_separator
        return text

    def multiple_text_to_annotations(self, text: str) -> List[Tuple[str]]:
        annotations = []
        for text in text.split(self.annotations_separator):
            ann = self.text_to_annotations(text)
            if ann != ():
                annotations.append(ann)
        return annotations


if __name__ == "__main__":
    class Args:
        pass

    def test_paraphrase():
        absa_data = AbsaData()
        args = Args()
        args.task_type = TaskType.T2A
        args.absa_tuple = AbsaTupleType.ACOP
        former = ParaphraseAnnotationsTextFormer(absa_data, args)

        annotations = ("open sesame", "restaurant general", "positive", "come back")
        text = former.annotations_to_text(annotations)
        print("Text:", text)
        print("Ref Annotations:", annotations)

        annotations = former.text_to_annotations(text)
        print("Annotations:", annotations)


    test_paraphrase()

