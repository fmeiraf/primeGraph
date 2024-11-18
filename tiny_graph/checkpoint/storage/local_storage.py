# checkpoint_library/storage_backends/in_memory.py

import logging
from datetime import datetime
from typing import List, Optional, Type

from pydantic import BaseModel

from tiny_graph.checkpoint.serialization import deserialize_model, serialize_model
from tiny_graph.checkpoint.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class LocalStorage(StorageBackend):
    def save_checkpoint(
        self,
        model_instance: BaseModel,
        chain_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        checkpoint_id = self._enforce_checkpoint_id(checkpoint_id)

        if not self._enforce_same_model_version(model_instance, chain_id):
            raise ValueError(
                "Model version mismatch: stored model version is different from current model version."
            )

        serialized_data = serialize_model(model_instance)
        with self._lock:
            self._storage[chain_id][checkpoint_id] = {
                "chain_id": chain_id,
                "model_class": model_instance.__class__,
                "model_version": getattr(model_instance, "version", None),
                "data": serialized_data,
                "timestamp": datetime.now(),
            }
        logger.info(f"Checkpoint '{checkpoint_id}' saved in memory.")
        return checkpoint_id

    def load_checkpoint(
        self, model_class: Type[BaseModel], chain_id: str, checkpoint_id: str
    ) -> BaseModel:
        with self._lock:
            chain_storage = self._storage.get(chain_id, {})
            if not chain_storage:
                raise KeyError(f"Chain '{chain_id}' not found.")
            checkpoint = chain_storage.get(checkpoint_id)
            if not checkpoint:
                raise KeyError(f"Checkpoint '{checkpoint_id}' not found.")

            # Check version compatibility
            stored_version = self._get_last_stored_model_version(chain_id)
            current_version = getattr(model_class, "version", None)
            if stored_version != current_version:
                raise ValueError(
                    f"Schema version mismatch: stored version is {stored_version}, "
                    f"but current model version is {current_version}."
                )
            return deserialize_model(model_class, checkpoint["data"])

    def list_checkpoints(self, chain_id: str) -> List[str]:
        with self._lock:
            chain_storage = self._storage.get(chain_id, {})
            if not chain_storage:
                return []
            return [name for name in chain_storage.keys()]

    def delete_checkpoint(self, checkpoint_id: str, chain_id: str) -> None:
        with self._lock:
            chain_storage = self._storage.get(chain_id, {})
            if not chain_storage:
                raise KeyError(f"Chain '{chain_id}' not found.")
            if checkpoint_id in chain_storage:
                del chain_storage[checkpoint_id]
                logger.info(f"Checkpoint '{checkpoint_id}' deleted from memory.")
            else:
                raise KeyError(f"Checkpoint '{checkpoint_id}' not found.")
