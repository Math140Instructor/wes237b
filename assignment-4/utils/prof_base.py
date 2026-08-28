from abc import ABC, abstractmethod
from typing import List

class ProfBase(ABC):
    @abstractmethod
    def profile(self) -> List[str]:
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self) -> float:
        raise NotImplementedError