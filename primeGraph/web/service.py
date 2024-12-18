import logging
from datetime import datetime
from typing import Dict, Set

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from primeGraph.checkpoint.base import StorageBackend
from primeGraph.graph.executable import Graph

from .models import ExecutionRequest, ExecutionResponse, GraphStatus

logger = logging.getLogger(__name__)


# TODO: Add support for sharing graph metadata
class GraphService:
  def __init__(
    self,
    graph: Graph,
    checkpoint_storage: StorageBackend,
    path_prefix: str = "/graph",
  ):
    self.router = APIRouter(prefix=path_prefix)
    self.graph = graph
    self.checkpoint_storage = checkpoint_storage
    self.active_websockets: Dict[str, Set[WebSocket]] = {}

    self._setup_routes()
    self._setup_websocket()
    self.graph.event_handlers.append(self._handle_graph_event)

  def _setup_routes(self) -> None:
    @self.router.post("/start")
    async def start_execution(request: ExecutionRequest) -> ExecutionResponse:
      try:
        chain_id = await self.graph.start_async(chain_id=request.chain_id, timeout=request.timeout)
        return self._create_response(chain_id)
      except Exception as e:
        logger.error(f"Error starting execution: {e!s}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    @self.router.post("/resume")
    async def resume_execution(request: ExecutionRequest) -> ExecutionResponse:
      try:
        await self.graph.resume_async(start_from=request.start_from)
        return self._create_response(self.graph.chain_id)
      except Exception as e:
        logger.error(f"Error resuming execution: {e!s}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    @self.router.get("/status/{chain_id}")
    async def get_status(chain_id: str) -> GraphStatus:
      try:
        if not self.graph.state:
          raise ValueError("State is not initialized")

        last_checkpoint_id = self.checkpoint_storage.get_last_checkpoint_id(chain_id)
        if not last_checkpoint_id:
          raise ValueError("No checkpoint found")

        checkpoint = self.checkpoint_storage.load_checkpoint(
          state_instance=self.graph.state,
          chain_id=chain_id,
          checkpoint_id=last_checkpoint_id,
        )
        return GraphStatus(
          chain_id=chain_id,
          status=checkpoint.chain_status,
          current_node=checkpoint.next_execution_node,
          executed_nodes=checkpoint.executed_nodes or set(),
          last_update=checkpoint.timestamp,
        )
      except Exception as e:
        logger.error(f"Error getting status: {e!s}")
        raise HTTPException(status_code=404, detail=str(e)) from e

  def _setup_websocket(self) -> None:
    @self.router.websocket("/ws/{chain_id}")
    async def websocket_endpoint(websocket: WebSocket, chain_id: str) -> None:
      logger.debug(f"WebSocket connection attempt for chain_id: {chain_id}")
      await websocket.accept()
      logger.debug(f"WebSocket connection accepted for chain_id: {chain_id}")

      if chain_id not in self.active_websockets:
        self.active_websockets[chain_id] = set()
      self.active_websockets[chain_id].add(websocket)
      logger.debug(f"Added websocket to active connections for chain_id: {chain_id}")

      try:
        while True:
          data = await websocket.receive_text()
          logger.debug(f"Received WebSocket message: {data}")
      except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected for chain_id: {chain_id}")
        self.active_websockets[chain_id].remove(websocket)
        if not self.active_websockets[chain_id]:
          del self.active_websockets[chain_id]

  def _create_response(self, chain_id: str) -> ExecutionResponse:
    return ExecutionResponse(
      chain_id=chain_id,
      status=self.graph.chain_status,
      next_execution_node=self.graph.next_execution_node,
      executed_nodes=self.graph.executed_nodes,
      timestamp=datetime.now(),
    )

  async def broadcast_status_update(self, chain_id: str) -> None:
    """Broadcast status updates to all connected WebSocket clients"""
    if chain_id in self.active_websockets:
      # Convert all data to JSON-serializable format
      status_data = {
        "type": "status",
        "chain_id": chain_id,
        "status": self.graph.chain_status.value,  # Convert enum to string
        "current_node": self.graph.next_execution_node,
        "executed_nodes": list(self.graph.executed_nodes) if self.graph.executed_nodes else [],  # Convert set to list
        "last_update": datetime.now().isoformat(),  # Convert datetime to string
      }

      # Create a copy of the set to safely iterate over
      websockets = self.active_websockets[chain_id].copy()
      disconnected = set()

      # Broadcast to all connected clients for this chain
      for websocket in websockets:
        try:
          await websocket.send_json(status_data)  # Send the dictionary directly
        except Exception as e:
          logger.error(f"Error broadcasting to websocket: {e!s}")
          disconnected.add(websocket)

      # Remove disconnected websockets after iteration
      if disconnected:
        self.active_websockets[chain_id] -= disconnected
        if not self.active_websockets[chain_id]:
          del self.active_websockets[chain_id]

  async def _handle_graph_event(self, event: dict) -> None:
    if event["chain_id"] in self.active_websockets:
      # Convert datetime objects to ISO format strings in the event dictionary
      serializable_event = event.copy()
      for key, value in serializable_event.items():
        if isinstance(value, datetime):
          serializable_event[key] = value.isoformat()

      # Create a copy of the set to safely iterate over
      websockets = self.active_websockets[event["chain_id"]].copy()
      disconnected = set()

      for websocket in websockets:
        try:
          await websocket.send_json(serializable_event)
        except Exception as e:
          logger.error(f"Error sending event: {e!s}")
          disconnected.add(websocket)

      # Remove disconnected websockets after iteration
      if disconnected:
        self.active_websockets[event["chain_id"]] -= disconnected
        if not self.active_websockets[event["chain_id"]]:
          del self.active_websockets[event["chain_id"]]


def create_graph_service(
  graph: Graph,
  checkpoint_storage: StorageBackend,
  path_prefix: str = "/graph",
) -> GraphService:
  """Factory function to create a new GraphService instance"""
  return GraphService(graph, checkpoint_storage, path_prefix)
