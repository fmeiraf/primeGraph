import logging
from datetime import datetime
from typing import Dict, Set

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from tiny_graph.checkpoint.base import StorageBackend
from tiny_graph.graph.executable import Graph

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

    def _setup_routes(self):
        @self.router.post("/start")
        async def start_execution(request: ExecutionRequest) -> ExecutionResponse:
            try:
                chain_id = await self.graph.start_async(
                    chain_id=request.chain_id, timeout=request.timeout
                )
                return self._create_response(chain_id)
            except Exception as e:
                logger.error(f"Error starting execution: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/resume")
        async def resume_execution(request: ExecutionRequest) -> ExecutionResponse:
            try:
                await self.graph.resume_async(start_from=request.start_from)
                return self._create_response(self.graph.chain_id)
            except Exception as e:
                logger.error(f"Error resuming execution: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/status/{chain_id}")
        async def get_status(chain_id: str) -> GraphStatus:
            try:
                checkpoint = self.checkpoint_storage.load_checkpoint(
                    state_instance=self.graph.state,
                    chain_id=chain_id,
                    checkpoint_id=self.checkpoint_storage.get_last_checkpoint_id(
                        chain_id
                    ),
                )
                return GraphStatus(
                    chain_id=chain_id,
                    status=checkpoint.chain_status,
                    current_node=checkpoint.next_execution_node,
                    executed_nodes=checkpoint.executed_nodes or set(),
                    last_update=checkpoint.timestamp,
                )
            except Exception as e:
                logger.error(f"Error getting status: {str(e)}")
                raise HTTPException(status_code=404, detail=str(e))

    def _setup_websocket(self):
        @self.router.websocket("/ws/{chain_id}")
        async def websocket_endpoint(websocket: WebSocket, chain_id: str):  # noqa
            await websocket.accept()

            if chain_id not in self.active_websockets:
                self.active_websockets[chain_id] = set()
            self.active_websockets[chain_id].add(websocket)

            try:
                while True:
                    # Keep connection alive and handle any incoming messages
                    data = await websocket.receive_text()  # noqa
                    # Handle any incoming messages if needed
                    # TODO: Implement message handling
            except WebSocketDisconnect:
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

    async def broadcast_status_update(self, chain_id: str):
        """Broadcast status updates to all connected WebSocket clients"""
        if chain_id in self.active_websockets:
            status = GraphStatus(
                chain_id=chain_id,
                status=self.graph.chain_status,
                current_node=self.graph.next_execution_node,
                executed_nodes=self.graph.executed_nodes,
                last_update=datetime.now(),
            )

            # Broadcast to all connected clients for this chain
            for websocket in self.active_websockets[chain_id]:
                try:
                    await websocket.send_json(status.model_dump())
                except Exception as e:
                    logger.error(f"Error broadcasting to websocket: {str(e)}")
                    # Remove failed websocket
                    self.active_websockets[chain_id].remove(websocket)


def create_graph_service(
    graph: Graph,
    checkpoint_storage: StorageBackend,
    path_prefix: str = "/graph",
) -> GraphService:
    """Factory function to create a new GraphService instance"""
    return GraphService(graph, checkpoint_storage, path_prefix)
