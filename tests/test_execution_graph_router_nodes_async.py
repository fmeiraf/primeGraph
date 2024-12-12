import asyncio
import time

import pytest

from tiny_graph.buffer.factory import History, LastValue
from tiny_graph.constants import END, START
from tiny_graph.graph.executable import Graph
from tiny_graph.models.state import GraphState


class RouterState(GraphState):
    result: LastValue[dict]  # Store the result from routes
    execution_order: History[str]  # Track execution order


@pytest.mark.asyncio
async def test_simple_router_async():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def process_data(state):
        if True:
            return "route_a"  # Router node returns next node name
        else:
            return "route_b"

    @graph.node()
    async def route_a(state):
        await asyncio.sleep(0.1)
        return {
            "result": {"path": "A"},
            "execution_order": "route_a",
        }

    @graph.node()
    async def route_b(state):
        await asyncio.sleep(0.1)
        return {
            "result": {"path": "B"},
            "execution_order": "route_b",
        }

    # Add router edge and possible routes
    graph.add_router_edge(START, "process_data")
    graph.add_edge("route_a", END)
    graph.add_edge("route_b", END)

    graph.compile()
    await graph.start_async()

    # Verify execution followed route_a
    assert state.result == {"path": "A"}
    assert state.execution_order == ["route_a"]


@pytest.mark.asyncio
async def test_complex_router_paths_async():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def initial_router(state):
        if True:
            return "path_b"
        else:
            return "path_a"

    @graph.node()
    async def path_a(state):
        await asyncio.sleep(0.1)
        return {
            "result": {"step": 1, "path": "A"},
            "execution_order": "path_a",
        }

    @graph.node()
    async def path_b(state):
        await asyncio.sleep(0.1)
        return {
            "result": {"step": 1, "path": "B"},
            "execution_order": "path_b",
        }

    @graph.node()
    async def secondary_router(state):
        if True:
            return "final_b"
        else:
            return "final_a"

    @graph.node()
    async def final_a(state):
        await asyncio.sleep(0.1)
        return {
            "result": {"step": 2, "path": "A-Final"},
            "execution_order": "final_a",
        }

    @graph.node()
    async def final_b(state):
        await asyncio.sleep(0.1)
        return {
            "result": {"step": 2, "path": "B-Final"},
            "execution_order": "final_b",
        }

    # Build graph with multiple routers
    graph.add_router_edge(START, "initial_router")
    graph.add_router_edge("path_a", "secondary_router")
    graph.add_router_edge("path_b", "secondary_router")
    graph.add_edge("final_a", END)
    graph.add_edge("final_b", END)

    graph.compile()
    await graph.start_async()

    # Verify execution followed path_b -> final_b
    assert state.result == {"step": 2, "path": "B-Final"}
    assert state.execution_order == ["path_b", "final_b"]


@pytest.mark.asyncio
async def test_parallel_router_execution_async():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def start_router(state):
        if True:
            return "parallel_a"
        else:
            return "parallel_b"

    @graph.node()
    async def parallel_a(state):
        await asyncio.sleep(0.2)
        return {
            "result": {"path": "A"},
            "execution_order": "parallel_a",
        }

    @graph.node()
    async def parallel_b(state):
        await asyncio.sleep(0.2)
        return {
            "result": {"path": "B"},
            "execution_order": "parallel_b",
        }

    @graph.node()
    async def router_merge(state):
        return "final"

    @graph.node()
    async def final(state):
        return {
            "result": {"path": "Final"},
            "execution_order": "final",
        }

    # Create parallel paths with routers
    graph.add_router_edge(START, "start_router")
    graph.add_router_edge("parallel_a", "router_merge")
    graph.add_router_edge("parallel_b", "router_merge")
    graph.add_edge("final", END)

    graph.compile()

    start_time = time.time()
    await graph.start_async()
    execution_time = time.time() - start_time

    # Verify execution path
    assert state.result == {"path": "Final"}
    assert state.execution_order == ["parallel_a", "final"]

    # Verify execution time is close to single sleep duration
    assert execution_time < 0.3  # Should be close to 0.2s if properly parallel


@pytest.mark.asyncio
async def test_router_error_handling_async():
    graph = Graph()

    @graph.node()
    async def invalid_router():
        return 123  # Invalid return type

    @graph.node()
    async def route_a():
        return None

    with pytest.raises(ValueError):
        graph.add_router_edge(START, "invalid_router")


@pytest.mark.asyncio
async def test_router_with_nonexistent_route_async():
    graph = Graph()

    @graph.node()
    async def bad_router():
        return "nonexistent_route"

    @graph.node()
    async def route_a():
        return None

    with pytest.raises(
        ValueError,
    ):
        graph.add_router_edge(START, "bad_router")
