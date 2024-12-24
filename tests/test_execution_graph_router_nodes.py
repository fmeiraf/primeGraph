import time

import pytest

from primeGraph.buffer.factory import History, LastValue
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState


class RouterState(GraphState):
    result: LastValue[dict]  # Store the result from routes
    execution_order: History[str]  # Track execution order


def test_simple_router():
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
    graph.start()

    # Verify execution followed route_a
    assert state.result == {"path": "A"}
    assert state.execution_order == ["route_a"]


def test_complex_router_paths():
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
    graph.start()

    # Verify execution followed path_b -> final_b
    assert state.result == {"step": 2, "path": "B-Final"}
    assert state.execution_order == ["path_b", "final_b"]


def test_router_error_handling():
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


def test_router_with_nonexistent_route():
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


def test_nested_router_paths():
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
    graph.start()

    # Verify execution followed the path: process_data -> route_b -> route_b2 -> route_c -> route_d
    assert state.result == {"result": "from route D"}
    assert state.execution_order == ["route_b", "route_c", "route_d"]


# TODO: create a warning on compile to advise users on cyclical paths
def test_cyclical_router():
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
    graph.start()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b"]

    # First resume - should execute route_c and pause at route_b
    graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b"]

    # Second resume - should execute route_c and pause at route_b again
    graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b", "route_b"]

    # Third resume - pattern continues
    graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == [
        "route_a",
        "route_b",
        "route_b",
        "route_b",
        "route_b",
    ]


def test_cyclical_router_interrupt_before():
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
    graph.start()
    assert state.result == {"result": "from route A"}  # Empty because we pause before route_b
    assert state.execution_order == ["route_a"]

    # First resume - should execute route_b and pause before route_b again
    graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b"]

    # Second resume - should execute route_b and pause before route_b again
    graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b"]

    # Third resume - pattern continues
    graph.resume()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b", "route_b"]
