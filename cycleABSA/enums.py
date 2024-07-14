# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Enums

import enum


class Task(enum.Enum):
    T2A = "T2A"     # Text to aspects
    A2T = "A2T"     # Aspects to text


class AbsaTuple(enum.Enum):
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
