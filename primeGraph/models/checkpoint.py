from datetime import datetime
from typing import Any, Dict, Optional

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
    engine_state: Optional[Dict[str, Any]] = None
