from datetime import datetime
from typing import Optional, Set

from pydantic import BaseModel

from primeGraph.types import ChainStatus


class ExecutionRequest(BaseModel):
  chain_id: Optional[str] = None
  start_from: Optional[str] = None
  timeout: Optional[float] = None


class ExecutionResponse(BaseModel):
  chain_id: str
  status: ChainStatus
  next_execution_node: Optional[str] = None
  executed_nodes: Set[str]
  timestamp: datetime


class GraphStatus(BaseModel):
  chain_id: str
  status: ChainStatus
  current_node: Optional[str] = None
  executed_nodes: Set[str]
  last_update: datetime
