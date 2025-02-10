from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from primeGraph.types import ChainStatus


class ExecutionRequest(BaseModel):
    chain_id: Optional[str] = None
    start_from: Optional[str] = None
    timeout: Optional[float] = None


class ExecutionResponse(BaseModel):
    chain_id: str
    status: ChainStatus
    timestamp: datetime


class GraphStatus(BaseModel):
    chain_id: str
    status: ChainStatus
    last_update: datetime
