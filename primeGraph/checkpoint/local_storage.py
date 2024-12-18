# checkpoint_library/storage_backends/in_memory.py

import logging
from datetime import datetime
from typing import List, Optional

from primeGraph.checkpoint.base import CheckpointData, StorageBackend
from primeGraph.checkpoint.serialization import serialize_model
from primeGraph.models.checkpoint import Checkpoint
from primeGraph.models.state import GraphState

logger = logging.getLogger(__name__)


class LocalStorage(StorageBackend):
  def __init__(self) -> None:
    super().__init__()

  def save_checkpoint(
    self,
    state_instance: GraphState,
    checkpoint_data: CheckpointData,
  ) -> str:
    checkpoint_id = self._enforce_checkpoint_id(checkpoint_data.checkpoint_id)

    self._enforce_same_model_version(state_instance, checkpoint_data.chain_id)

    # Convert state class to string representation
    state_class_str = f"{state_instance.__class__.__module__}.{state_instance.__class__.__name__}"

    serialized_data = serialize_model(state_instance)
    with self._lock:
      self._storage[checkpoint_data.chain_id][checkpoint_id] = Checkpoint(
        checkpoint_id=checkpoint_id,
        chain_id=checkpoint_data.chain_id,
        chain_status=checkpoint_data.chain_status,
        state_class=state_class_str,
        state_version=getattr(state_instance, "version", None),
        data=serialized_data,
        timestamp=datetime.now(),
        next_execution_node=checkpoint_data.next_execution_node,
        executed_nodes=checkpoint_data.executed_nodes,
      )
    logger.info(f"Checkpoint '{checkpoint_id}' saved in memory.")
    return checkpoint_id

  def load_checkpoint(self, state_instance: GraphState, chain_id: str, checkpoint_id: str) -> Checkpoint:
    self._enforce_same_model_version(state_instance, chain_id)

    with self._lock:
      chain_storage = self._storage.get(chain_id, None)
      if not chain_storage:
        raise KeyError(f"Chain '{chain_id}' not found.")

      checkpoint = chain_storage.get(checkpoint_id, None)
      if not checkpoint:
        raise KeyError(f"Checkpoint '{checkpoint_id}' not found.")

      return checkpoint

  def list_checkpoints(
    self,
    chain_id: str,
  ) -> List[Checkpoint]:
    with self._lock:
      chain_storage = self._storage.get(chain_id, {})
      if not chain_storage:
        return []
      return list(chain_storage.values())

  def delete_checkpoint(self, chain_id: str, checkpoint_id: str) -> None:
    with self._lock:
      chain_storage = self._storage.get(chain_id, {})
      if not chain_storage:
        raise KeyError(f"Chain '{chain_id}' not found.")
      if checkpoint_id in chain_storage:
        del chain_storage[checkpoint_id]
        logger.info(f"Checkpoint '{checkpoint_id}' deleted from memory.")
      else:
        raise KeyError(f"Checkpoint '{checkpoint_id}' not found.")

  def get_last_checkpoint_id(self, chain_id: str) -> Optional[str]:
    with self._lock:
      chain_storage = self._storage.get(chain_id, {})
      return list(chain_storage.keys())[-1] if chain_storage else None
