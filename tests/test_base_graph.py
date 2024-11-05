from typing import NamedTuple

import pytest
from pydantic import BaseModel

from tiny_graph.constants import END, START
from tiny_graph.graph.base import Edge, Graph


# Test fixtures and helper classes
class _TestState(BaseModel):
    value: int
    name: str


class _TestNamedTupleState(NamedTuple):
    value: int
    name: str


@pytest.fixture
def empty_graph():
    return Graph()


@pytest.fixture
def basic_graph():
    graph = Graph()

    @graph.node()
    def start():
        pass

    @graph.node()
    def end():
        pass

    graph.add_edge(START, "start")
    graph.add_edge("start", "end")
    graph.add_edge("end", END)
    return graph


@pytest.fixture
def invalid_graph():
    graph = Graph()

    @graph.node()
    def start():
        pass

    @graph.node()
    def intermediate():
        pass

    @graph.node()
    def orphan():
        pass

    @graph.node()
    def end():
        pass

    graph.add_edge(START, "intermediate")
    graph.add_edge("intermediate", END)
    graph.add_edge(START, END)

    return graph


# Test basic graph creation and properties
def test_graph_initialization():
    graph = Graph()
    assert graph.edges == set()
    assert not graph.is_compiled
    assert START in graph.nodes
    assert END in graph.nodes


def test_graph_with_pydantic_state():
    state = _TestState(value=1, name="test")
    graph = Graph(state)
    assert graph.state_schema == {"value": int, "name": str}


def test_graph_with_namedtuple_state():
    state = _TestNamedTupleState(1, "test")
    graph = Graph(state)
    assert graph.state_schema == {"value": int, "name": str}


# Test node decoration and edge creation
def test_node_decoration(empty_graph):
    @empty_graph.node()
    def test_node():
        pass

    assert "test_node" in empty_graph.nodes
    assert empty_graph.nodes["test_node"].name == "test_node"
    assert not empty_graph.nodes["test_node"].is_async
    assert not empty_graph.nodes["test_node"].is_router


def test_node_with_custom_name(empty_graph):
    @empty_graph.node(name="custom")
    def test_node():
        pass

    assert "custom" in empty_graph.nodes
    assert "test_node" not in empty_graph.nodes


def test_node_with_metadata(empty_graph):
    metadata = {"key": "value"}

    @empty_graph.node(metadata=metadata)
    def test_node():
        pass

    assert empty_graph.nodes["test_node"].metadata == metadata


# Test router nodes
def test_router_node(empty_graph):
    @empty_graph.node()
    def route_a():
        pass

    @empty_graph.node()
    def route_b():
        pass

    @empty_graph.node()
    def router():
        return "route_a"

    empty_graph.add_edge("router", "route_a")
    empty_graph.add_edge("router", "route_b")

    assert empty_graph.nodes["router"].is_router
    assert "route_a" in empty_graph.nodes["router"].possible_routes


# Test compilation and validation
def test_compile_valid_graph(basic_graph):
    compiled = basic_graph.compile()
    assert compiled.is_compiled
    assert len(compiled.tasks) > 0


def test_compile_invalid_router():
    graph = Graph()

    @graph.node()
    def router():
        return "nonexistent"

    @graph.node()
    def actual():
        pass

    graph.add_edge("router", "actual")

    with pytest.raises(
        ValueError, match="Router node 'router' contains invalid routes"
    ):
        graph.compile()


def test_cannot_add_nodes_after_compile(basic_graph):
    basic_graph.compile()

    with pytest.raises(ValueError, match="Cannot add nodes after compiling"):

        @basic_graph.node()
        def new_node():
            pass


# Test async nodes
def test_async_node(empty_graph):
    @empty_graph.node()
    async def async_node():
        pass

    assert empty_graph.nodes["async_node"].is_async


# Test edge creation
def test_edge_creation(basic_graph):
    assert Edge("start", "end") in basic_graph.edges
    assert ("start", "end") in basic_graph._all_edges


def test_node_list(basic_graph):
    assert set(basic_graph._all_nodes) == {START, "start", "end", END}


def test_validate_graph(invalid_graph, basic_graph):
    assert basic_graph.validate()
    with pytest.raises(ValueError):
        invalid_graph.validate()


def test_validate_missing_start_node(empty_graph):
    @empty_graph.node()
    def end():
        pass

    with pytest.raises(ValueError):
        empty_graph.validate()


def test_validate_missing_end_node(empty_graph):
    @empty_graph.node()
    def start():
        pass

    with pytest.raises(ValueError):
        empty_graph.validate()


def test_validate_unreachable_end_node(empty_graph):
    @empty_graph.node()
    def start():
        pass

    @empty_graph.node()
    def end():
        pass

    # No edge connecting start to end
    with pytest.raises(ValueError):
        empty_graph.validate()


def test_validate_cycle_detection(empty_graph):
    @empty_graph.node()
    def start():
        pass

    @empty_graph.node()
    def node_a():
        pass

    @empty_graph.node()
    def node_b():
        pass

    @empty_graph.node()
    def end():
        pass

    # Create a cycle
    empty_graph.add_edge("start", "node_a")
    empty_graph.add_edge("node_a", "node_b")
    empty_graph.add_edge("node_b", "node_a")  # Creates cycle
    empty_graph.add_edge("node_b", "end")

    with pytest.raises(ValueError):
        empty_graph.validate()


def test_validate_orphaned_nodes(empty_graph):
    @empty_graph.node()
    def start():
        pass

    @empty_graph.node()
    def end():
        pass

    @empty_graph.node()
    def orphaned():
        pass

    empty_graph.add_edge(START, "start")
    empty_graph.add_edge("start", "end")
    empty_graph.add_edge("end", END)

    with pytest.raises(ValueError):
        empty_graph.validate()
