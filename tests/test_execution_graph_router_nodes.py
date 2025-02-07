import asyncio
import time

import pytest

from primeGraph.buffer.factory import History, LastValue
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState
from primeGraph.types import ChainStatus


class RouterState(GraphState):
    result: LastValue[dict]  # Store the result from routes
    execution_order: History[str]  # Track execution order


@pytest.mark.asyncio
async def test_simple_router():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def process_data(state):
        if True:
            return "route_a"  # Router node returns next node name
        else:
            return "route_b"

    @graph.node()
    def route_a(state):
        time.sleep(0.1)
        return {
            "result": {"path": "A"},
            "execution_order": "route_a",
        }

    @graph.node()
    def route_b(state):
        time.sleep(0.1)
        return {
            "result": {"path": "B"},
            "execution_order": "route_b",
        }

    # Add router edge and possible routes
    graph.add_router_edge(START, "process_data")
    graph.add_edge("route_a", END)
    graph.add_edge("route_b", END)

    graph.compile()
    await graph.execute()

    # Verify execution followed route_a
    assert state.result == {"path": "A"}
    assert state.execution_order == ["route_a"]


@pytest.mark.asyncio
async def test_complex_router_paths():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def initial_router(state):
        if True:
            return "path_b"
        else:
            return "path_a"

    @graph.node()
    def path_a(state):
        return {
            "result": {"step": 1, "path": "A"},
            "execution_order": "path_a",
        }

    @graph.node()
    def path_b(state):
        return {
            "result": {"step": 1, "path": "B"},
            "execution_order": "path_b",
        }

    @graph.node()
    def secondary_router(state):
        if True:
            return "final_b"
        else:
            return "final_a"

    @graph.node()
    def final_a(state):
        return {
            "result": {"step": 2, "path": "A-Final"},
            "execution_order": "final_a",
        }

    @graph.node()
    def final_b(state):
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
    await graph.execute()

    # Verify execution followed path_b -> final_b
    assert state.result == {"step": 2, "path": "B-Final"}
    assert state.execution_order == ["path_b", "final_b"]


@pytest.mark.asyncio
async def test_router_error_handling():
    graph = Graph()

    @graph.node()
    def invalid_router():
        return 123  # Invalid return type

    @graph.node()
    def route_a():
        return None

    # Test invalid router return type
    with pytest.raises(ValueError):
        graph.add_router_edge(START, "invalid_router")


@pytest.mark.asyncio
async def test_router_with_nonexistent_route():
    graph = Graph()

    @graph.node()
    def bad_router():
        return "nonexistent_route"

    @graph.node()
    def route_a():
        return None

    # Test routing to nonexistent node
    with pytest.raises(ValueError):
        graph.add_router_edge(START, "bad_router")


@pytest.mark.asyncio
async def test_nested_router_paths():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def process_data(state):
        if True:
            return "route_b"
        else:
            return "route_a"

    @graph.node()
    def route_a(state):
        return {
            "result": {"result": "from route A"},
            "execution_order": "route_a",
        }

    @graph.node()
    def route_b(state):
        return {
            "result": {"result": "from route B"},
            "execution_order": "route_b",
        }

    @graph.node()
    def route_a2(state):
        return {
            "result": {"result": "from route A2"},
            "execution_order": "route_a2",
        }

    @graph.node()
    def route_b2(state):
        return "route_c"

    @graph.node()
    def route_c(state):
        return {
            "result": {"result": "from route C"},
            "execution_order": "route_c",
        }

    @graph.node()
    def route_d(state):
        return {
            "result": {"result": "from route D"},
            "execution_order": "route_d",
        }

    # Add edges matching the notebook structure
    graph.add_router_edge(START, "process_data")
    graph.add_edge("route_a", "route_a2")
    graph.add_edge("route_a2", "route_c")
    graph.add_router_edge("route_b", "route_b2")
    graph.add_edge("route_c", "route_d")
    graph.add_edge("route_d", END)

    graph.compile()
    await graph.execute()

    # Verify execution followed the path: process_data -> route_b -> route_b2 -> route_c -> route_d
    assert state.result == {"result": "from route D"}
    assert state.execution_order == ["route_b", "route_c", "route_d"]


# TODO: create a warning on compile to advise users on cyclical paths
@pytest.mark.asyncio
async def test_cyclical_router():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def route_a(state):
        print("Executing route_a")
        return {"result": {"result": "from route A"}, "execution_order": "route_a"}

    @graph.node(interrupt="after")
    def route_b(state):
        print("Executing route_b")
        return {"result": {"result": "from route B"}, "execution_order": "route_b"}

    @graph.node()
    def route_c(state):
        print("Executing route_c")
        if True:
            return "route_b"
        return "route_d"

    @graph.node()
    def route_d(state):
        print("Executing route_d")
        return {"result": {"result": "from route D"}, "execution_order": "route_d"}

    # Add edges
    # graph.add_edge(START, "process_data")
    graph.add_edge(START, "route_a")  # No need to specify routes
    graph.add_edge("route_a", "route_b")
    graph.add_router_edge("route_b", "route_c")
    graph.add_edge("route_d", END)

    graph.compile()

    # Initial execution
    await graph.execute()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b"]

    # First resume - should execute route_c and pause at route_b
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b"]

    # Second resume - should execute route_c and pause at route_b again
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b", "route_b"]

    # Third resume - pattern continues
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == [
        "route_a",
        "route_b",
        "route_b",
        "route_b",
        "route_b",
    ]


@pytest.mark.asyncio
async def test_cyclical_router_interrupt_before():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def route_a(state):
        print("Executing route_a")
        return {"result": {"result": "from route A"}, "execution_order": "route_a"}

    @graph.node(interrupt="before")
    def route_b(state):
        print("Executing route_b")
        return {"result": {"result": "from route B"}, "execution_order": "route_b"}

    @graph.node()
    def route_c(state):
        print("Executing route_c")
        if True:
            return "route_b"
        return "route_d"

    @graph.node()
    def route_d(state):
        print("Executing route_d")
        return {"result": {"result": "from route D"}, "execution_order": "route_d"}

    # Add edges
    graph.add_edge(START, "route_a")
    graph.add_edge("route_a", "route_b")
    graph.add_router_edge("route_b", "route_c")
    graph.add_edge("route_d", END)

    graph.compile()

    # Initial execution - should pause before route_b
    await graph.execute()
    assert state.result == {"result": "from route A"}  # Empty because we pause before route_b
    assert state.execution_order == ["route_a"]

    # First resume - should execute route_b and pause before route_b again
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b"]

    # Second resume - should execute route_b and pause before route_b again
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b"]

    # Third resume - pattern continues
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b", "route_b"]


@pytest.mark.asyncio
async def test_chain_status_with_router():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def route_a(state):
        return {"result": {"result": "from route A"}, "execution_order": "route_a"}

    @graph.node()
    def route_b(state):
        return {"result": {"result": "from route B"}, "execution_order": "route_b"}

    @graph.node()
    def router(state):
        return "route_b"

    # Add edges
    graph.add_edge(START, "route_a")
    graph.add_router_edge("route_a", "router")
    graph.add_edge("route_b", END)

    graph.compile()
    await graph.execute()

    # Verify execution completed and chain status is DONE
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b"]
    assert graph.chain_status == ChainStatus.DONE


@pytest.mark.asyncio
async def test_chain_status_with_router_and_interrupts():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def route_a(state):
        return {"result": {"result": "from route A"}, "execution_order": "route_a"}

    @graph.node(interrupt="before")
    def route_b(state):
        return {"result": {"result": "from route B"}, "execution_order": "route_b"}

    @graph.node()
    def router(state):
        return "route_b"

    @graph.node()
    def route_c(state):
        return {"result": {"result": "from route C"}, "execution_order": "route_c"}

    # Add edges
    graph.add_edge(START, "route_a")
    graph.add_router_edge("route_a", "router")
    graph.add_edge("route_b", "route_c")
    graph.add_edge("route_c", END)

    graph.compile()

    # First execution - should stop before route_b
    await graph.execute()
    assert state.result == {"result": "from route A"}
    assert state.execution_order == ["route_a"]
    assert graph.chain_status == ChainStatus.PAUSE

    # Second execution - should stop after route_c
    await graph.resume()
    assert state.result == {"result": "from route C"}
    assert state.execution_order == ["route_a", "route_b", "route_c"]
    assert graph.chain_status == ChainStatus.DONE



@pytest.mark.asyncio
async def test_async_cyclical_router_interrupt_before():
    state = RouterState(result={}, execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def route_a(state):
        print("Executing route_a")
        return {"result": {"result": "from route A"}, "execution_order": "route_a"}

    @graph.node(interrupt="before")
    async def route_b(state):
        print("Executing route_b")
        return {"result": {"result": "from route B"}, "execution_order": "route_b"}

    @graph.node()
    async def route_c(state):
        print("Executing route_c")
        if True:
            return "route_b"
        return "route_d"

    @graph.node()
    async def route_d(state):
        print("Executing route_d")
        return {"result": {"result": "from route D"}, "execution_order": "route_d"}

    # Add edges
    graph.add_edge(START, "route_a")
    graph.add_edge("route_a", "route_b")
    graph.add_router_edge("route_b", "route_c")
    graph.add_edge("route_d", END)

    graph.compile()

    # Initial execution - should pause before route_b
    await graph.execute()
    assert state.result == {"result": "from route A"}
    assert state.execution_order == ["route_a"]

    # First resume - should execute route_b and pause before route_b again
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b"]

    # Second resume - should execute route_b and pause before route_b again
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b"]

    # Third resume - pattern continues
    await graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b", "route_b"]


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
    await graph.execute()
    execution_time = time.time() - start_time

    # Verify execution path
    assert state.result == {"path": "Final"}
    assert state.execution_order == ["parallel_a", "final"]

    # Verify execution time is close to single sleep duration
    assert execution_time < 0.3  # Should be close to 0.2s if properly parallel