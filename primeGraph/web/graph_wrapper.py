import asyncio
from asyncio import Task

from primeGraph.graph.executable import Graph

from .service import GraphService


def wrap_graph_with_websocket(graph: Graph, service: GraphService) -> Graph:
  """Wraps a Graph instance to add WebSocket broadcasting capabilities"""

  # Store original methods
  original_save_checkpoint = graph._save_checkpoint

  def sync_broadcast_node_completion() -> Task[None]:
    if hasattr(graph, "chain_id"):
      loop = asyncio.get_event_loop()
      task = loop.create_task(service.broadcast_status_update(graph.chain_id))
      return task  # Optionally store or handle the task reference
    else:
      raise ValueError("Graph instance does not have a chain_id attribute")

  async def async_broadcast_node_completion() -> None:
    if hasattr(graph, "chain_id"):
      await service.broadcast_status_update(graph.chain_id)

  # Override checkpoint method to include broadcasting
  def new_save_checkpoint(node_name: str) -> None:
    original_save_checkpoint(node_name)
    sync_broadcast_node_completion()

  # Replace the method
  graph._save_checkpoint = new_save_checkpoint  # type: ignore
  # Store async version for async contexts
  graph._async_save_checkpoint = async_broadcast_node_completion  # type: ignore

  return graph
