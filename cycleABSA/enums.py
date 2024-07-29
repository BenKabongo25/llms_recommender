# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Enums

import enum


class TaskType(enum.Enum):
    T2A = "T2A"     # Text to aspects
    A2T = "A2T"     # Aspects to text


class AbsaTupleType(enum.Enum):
    A = "a"         # aspect terms
    C = "c"         # aspect categories
    O = "o"         # opinion terms
    P = "p"         # sentiment polarities
    AC = "ac"
    AO = "ao"
    AP = "ap"
    CO = "co"
    CP = "cp"
    OP = "op"
    ACO = "aco"
    ACP = "acp"
    AOP = "aop"
    COP = "cop"
    ACOP = "acop"


    @classmethod
    def format_annotations(cls, annotations: tuple, raw_format: str, absa_tuple: str) -> tuple:
        if not isinstance(absa_tuple, str):
            absa_tuple = absa_tuple.value
        
        if absa_tuple == raw_format:
            return annotations

        formatted_annotations = []
        for element in absa_tuple:
            element_idx = raw_format.index(element)
            formatted_annotations.append(annotations[element_idx])
        return tuple(formatted_annotations)



class AnnotationsTextFormerType(enum.Enum):
    GAS_EXTRACTION_STYLE = "gas_extraction_style"
    GAS_ANNOTATION_STYLE = "gas_annotation_style"
    PARAPHRASE           = "paraphrase"


class PrompterType(enum.Enum):
    P1 = 1
    P2 = 2