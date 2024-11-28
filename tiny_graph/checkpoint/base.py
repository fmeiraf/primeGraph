import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Set

from tiny_graph.models.checkpoint import Checkpoint
from tiny_graph.models.state import GraphState
from tiny_graph.types import ChainStatus


class StorageBackend(ABC):
    def __init__(self):
        self._storage = defaultdict(dict)
        self._lock = threading.Lock()

    def _enforce_checkpoint_id(self, checkpoint_id: Optional[str]) -> str:
        return checkpoint_id or f"checkpoint_{uuid.uuid4()}"

    def _get_last_stored_model_version(self, chain_id: str) -> Optional[str]:
        chain_storage = self._storage.get(chain_id, None)
        if not chain_storage:
            return None
        sorted_checkpoints = sorted(chain_storage.values(), key=lambda x: x.timestamp)
        return sorted_checkpoints[-1].state_version if sorted_checkpoints else None

    def _enforce_same_model_version(
        self,
        state_instance: GraphState,
        chain_id: str,
    ) -> bool:
        current_version = getattr(state_instance, "version", None)
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
        state_instance: GraphState,
        chain_id: str,
        chain_status: ChainStatus,
        checkpoint_id: Optional[str] = None,
        next_execution_node: Optional[str] = None,
        executed_nodes: Optional[Set[str]] = None,
    ) -> str:
        pass

    @abstractmethod
    def load_checkpoint(
        self, state_instance: GraphState, chain_id: str, checkpoint_id: str
    ) -> Checkpoint:
        pass

    @abstractmethod
    def list_checkpoints(
        self,
        state_instance: Optional[GraphState] = None,
        chain_id: Optional[str] = None,
    ) -> List[Checkpoint]:
        pass

    @abstractmethod
    def delete_checkpoint(
        self, checkpoint_id: str, chain_id: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def get_last_checkpoint_id(self, chain_id: str) -> Optional[str]:
        pass
