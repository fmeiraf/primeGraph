from abc import ABC, abstractmethod
from typing import List, Optional, Type

from pydantic import BaseModel


class StorageBackend(ABC):
    @abstractmethod
    def save_checkpoint(
        self, model_instance: BaseModel, checkpoint_name: Optional[str] = None
    ) -> str:
        pass

    @abstractmethod
    def load_checkpoint(
        self, model_class: Type[BaseModel], checkpoint_name: str
    ) -> BaseModel:
        pass

    @abstractmethod
    def list_checkpoints(
        self, model_class: Optional[Type[BaseModel]] = None
    ) -> List[str]:
        pass

    @abstractmethod
    def delete_checkpoint(self, checkpoint_name: str) -> None:
        pass
