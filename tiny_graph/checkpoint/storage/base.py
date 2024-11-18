import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Type

from pydantic import BaseModel


class StorageBackend(ABC):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self._storage = defaultdict(dict)
        self._lock = threading.Lock()

    def _enforce_checkpoint_id(self, checkpoint_id: Optional[str]) -> str:
        return checkpoint_id or f"{self.chain_id}_{uuid.uuid4()}"

    def _get_last_stored_model_version(self, chain_id: str) -> Optional[str]:
        chain_storage = self._storage.get(chain_id, {})
        if not chain_storage:
            return None
        sorted_checkpoints = sorted(
            chain_storage.values(), key=lambda x: x["timestamp"]
        )
        return sorted_checkpoints[-1]["model_version"] if sorted_checkpoints else None

    def _enforce_same_model_version(
        self,
        model_class: Type[BaseModel],
        chain_id: str,
    ) -> bool:
        current_version = getattr(model_class, "version", None)
        stored_version = self._get_last_stored_model_version(chain_id)
        if not stored_version:
            return True

        if not current_version:
            raise ValueError(
                "Model version for current model is not set. "
                "Please set the 'version' attribute in the model."
            )
        if stored_version != current_version:
            raise ValueError(
                f"Schema version mismatch: stored version is {stored_version}, "
                f"but current model version is {current_version}."
            )
        else:
            return True

    @abstractmethod
    def save_checkpoint(
        self,
        model_instance: BaseModel,
        chain_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        pass

    @abstractmethod
    def load_checkpoint(
        self, model_class: Type[BaseModel], chain_id: str, checkpoint_id: str
    ) -> BaseModel:
        pass

    @abstractmethod
    def list_checkpoints(
        self,
        model_class: Optional[Type[BaseModel]] = None,
        chain_id: Optional[str] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def delete_checkpoint(
        self, checkpoint_id: str, chain_id: Optional[str] = None
    ) -> None:
        pass
