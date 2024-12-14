import asyncio

from primeGraph.graph.executable import Graph

from .service import GraphService


def wrap_graph_with_websocket(graph: Graph, service: GraphService):
    """Wraps a Graph instance to add WebSocket broadcasting capabilities"""

    # Store original methods
    original_save_checkpoint = graph._save_checkpoint

    def sync_broadcast_node_completion(node_name: str):
        if hasattr(graph, "chain_id"):
            loop = asyncio.get_event_loop()
            loop.create_task(service.broadcast_status_update(graph.chain_id))

    async def async_broadcast_node_completion(node_name: str):
        if hasattr(graph, "chain_id"):
            await service.broadcast_status_update(graph.chain_id)

    # Override checkpoint method to include broadcasting
    def new_save_checkpoint(node_name: str):
        original_save_checkpoint(node_name)
        sync_broadcast_node_completion(node_name)

    # Replace the method
    graph._save_checkpoint = new_save_checkpoint
    # Store async version for async contexts
    graph._async_save_checkpoint = async_broadcast_node_completion

    return graph
