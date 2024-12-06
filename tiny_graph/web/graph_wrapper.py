from tiny_graph.graph.executable import Graph
from tiny_graph.types import ChainStatus

from .service import GraphService


def wrap_graph_with_websocket(graph: Graph, service: GraphService):
    """Wraps a Graph instance to add WebSocket broadcasting capabilities"""
    original_update_chain_status = graph._update_chain_status

    def new_update_chain_status(status: ChainStatus):
        # Call original synchronously
        original_update_chain_status(status)

        # Create and run broadcast coroutine if chain_id exists
        if hasattr(graph, "chain_id"):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(service.broadcast_status_update(graph.chain_id))
            except RuntimeError:
                # Handle case where there's no running event loop
                asyncio.run(service.broadcast_status_update(graph.chain_id))

    graph._update_chain_status = new_update_chain_status
    return graph
