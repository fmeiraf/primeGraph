from tiny_graph.graph.executable import Graph

from .service import GraphService


def wrap_graph_with_websocket(graph: Graph, service: GraphService):
    """Wraps a Graph instance to add WebSocket broadcasting capabilities"""
    original_update_chain_status = graph._update_chain_status

    async def new_update_chain_status(status):
        original_update_chain_status(status)
        if hasattr(graph, "chain_id"):
            await service.broadcast_status_update(graph.chain_id)

    graph._update_chain_status = new_update_chain_status
    return graph
