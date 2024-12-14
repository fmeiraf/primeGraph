from datetime import datetime
from typing import Optional, Set

from pydantic import BaseModel

from primeGraph.types import ChainStatus


class Checkpoint(BaseModel):
    checkpoint_id: str
    chain_id: str
    chain_status: ChainStatus
    state_class: str  # Store as string to avoid serialization issues
    state_version: Optional[str] = None
    data: str  # Serialized state data
    timestamp: datetime
    next_execution_node: Optional[str] = None
    last_executed_node: Optional[str] = None
    executed_nodes: Optional[Set[str]] = None
