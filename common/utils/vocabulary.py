# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Common - Vocabulary

from tqdm import tqdm
from typing import *


class Vocabulary:

    def __init__(self):
        self._elements2ids = {}
        self._ids2elements = {}
        self.n_elements = 0

    def add_element(self, element: Union[int, float, str]):
        if element not in self._elements2ids:
            self.n_elements += 1
            self._elements2ids[element] = self.n_elements
            self._ids2elements[self.n_elements] = element

    def add_elements(self, elements: List[Union[int, float, str]]):
        for element in tqdm(elements, "Vocabulary creation", colour="green"):
            self.add_element(element)

    def __len__(self):
        return self.n_elements
    
    def id2element(self, id: int) -> Union[int, float, str]:
        return self._ids2elements[id]
    
    def element2id(self, element: Union[int, float, str]) -> int:
        return self._elements2ids[element]
    
    def ids2elements(self, ids: List[int]) -> List[Union[int, float, str]]:
        return [self._ids2elements[id] for id in ids]
    
    def elements2ids(self, elements: List[Union[int, float, str]]) -> List[int]:
        return [self._elements2ids[element] for element in elements]
