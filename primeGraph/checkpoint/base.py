import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from primeGraph.models.checkpoint import Checkpoint
from primeGraph.models.state import GraphState
from primeGraph.types import ChainStatus


@dataclass
class CheckpointData:
  chain_id: str
  chain_status: ChainStatus
  checkpoint_id: Optional[str] = None
  next_execution_node: Optional[str] = None
  executed_nodes: Optional[Set[str]] = None


class StorageBackend(ABC):
  def __init__(self) -> None:
    self._storage: Dict[str, Dict[str, Checkpoint]] = defaultdict(dict)
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
        "Model version for current model is not set. " "Please set the 'version' attribute in the model."
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
    checkpoint_data: CheckpointData,
  ) -> str:
    pass

  @abstractmethod
  def load_checkpoint(self, state_instance: GraphState, chain_id: str, checkpoint_id: str) -> Checkpoint:
    pass

  @abstractmethod
  def list_checkpoints(
    self,
    chain_id: str,
  ) -> List[Checkpoint]:
    pass

  @abstractmethod
  def delete_checkpoint(self, chain_id: str, checkpoint_id: str) -> None:
    pass

  @abstractmethod
  def get_last_checkpoint_id(self, chain_id: str) -> Optional[str]:
    pass
