from typing import NamedTuple

import pytest
from pydantic import BaseModel

from primeGraph.constants import END, START
from primeGraph.graph.base import BaseGraph, Edge


# Test fixtures and helper classes
class _TestState(BaseModel):
    value: int
    name: str


class _TestNamedTupleState(NamedTuple):
    value: int
    name: str


@pytest.fixture
def empty_graph():
    return BaseGraph()


@pytest.fixture
def basic_graph():
    graph = BaseGraph()

    @graph.node()
    def start():
        pass

    @graph.node()
    def end():
        pass

    graph.add_edge(START, "start")
    graph.add_edge("start", "end", "testing_edge")
    graph.add_edge("end", END)
    return graph


@pytest.fixture
def invalid_graph():
    graph = BaseGraph()

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


@pytest.fixture
def simple_fan_graph():
    graph = BaseGraph()

    @graph.node()
    def escape():
        print("Starting workflow")

    @graph.node()
    def process_data():
        print("Processing data")

    @graph.node()
    def validate():
        print("Validating results")

    @graph.node()
    def aa():
        print("Validating results")

    @graph.node()
    def bb():
        print("Validating results")

    @graph.node()
    def prep():
        print("Workflow complete")

    graph.add_edge(START, "process_data")
    graph.add_edge("process_data", "validate")
    graph.add_edge("validate", "escape")
    graph.add_edge("escape", "prep")
    graph.add_edge("validate", "aa")
    graph.add_edge("aa", "bb")
    graph.add_edge("bb", "prep")
    graph.add_edge("prep", END)

    return graph


@pytest.fixture
def complex_fan_graph():
    graph = BaseGraph()

    @graph.node()
    def escape():
        print("Starting workflow")

    @graph.node()
    def process_data():
        print("Processing data")

    @graph.node()
    def validate():
        print("Validating results")

    @graph.node()
    def aa():
        print("Validating results")

    @graph.node()
    def bb():
        print("Validating results")

    @graph.node()
    def dd():
        print("Validating results")

    @graph.node()
    def cc():
        print("Validating results")

    @graph.node()
    def hh():
        print("Validating results")

    @graph.node()
    def prep():
        print("Workflow complete")

    # Add edges to create workflow
    graph.add_edge(START, "process_data")
    graph.add_edge("process_data", "validate")
    graph.add_edge("validate", "escape")
    graph.add_edge("escape", "dd")
    graph.add_edge("escape", "cc")
    graph.add_edge("cc", "hh")
    graph.add_edge("dd", "hh")
    graph.add_edge("hh", "prep")
    graph.add_edge("validate", "aa")
    graph.add_edge("aa", "bb")
    graph.add_edge("bb", "prep")
    graph.add_edge("prep", END)

    return graph


# Test basic graph creation and properties
def test_graph_initialization():
    graph = BaseGraph()
    assert graph.edges == set()
    assert not graph.is_compiled
    assert START in graph.nodes
    assert END in graph.nodes


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


# Test compilation and validation
def test_compile_valid_graph(basic_graph):
    compiled = basic_graph.compile()
    assert compiled.is_compiled


def test_graph_with_only_one_node(empty_graph):
    shallow_graph = BaseGraph()

    @shallow_graph.node()
    def task1():
        pass

    shallow_graph.add_edge(START, "task1")
    shallow_graph.add_edge("task1", END)
    with pytest.raises(ValueError):
        shallow_graph.compile()


def test_compile_invalid_router():
    graph = BaseGraph()

    @graph.node()
    def router():
        return "nonexistent"

    @graph.node()
    def actual():
        pass

    graph.add_edge("router", "actual")

    with pytest.raises(ValueError):
        graph.compile()


def test_cannot_add_nodes_after_compile(basic_graph):
    basic_graph.compile()

    with pytest.raises(ValueError):

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
    assert Edge("start", "end", "testing_edge") in basic_graph.edges


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


def normalize_execution_plan(plan):
    """Recursively sort lists within the execution plan while preserving structure."""
    if not isinstance(plan, list):
        return plan

    # First normalize any nested lists
    normalized = [normalize_execution_plan(item) for item in plan]

    # Sort only lists that contain all strings or all lists
    if all(isinstance(x, str) for x in normalized) or all(
        isinstance(x, list) for x in normalized
    ):
        return sorted(normalized)
    return normalized


def get_length(item):
    if isinstance(item, list):
        return len(item)
    return 1


def get_item_type(item):
    if isinstance(item, list):
        return "list"
    return "str/other"


def create_compable_list(plan):
    return [(get_length(item), get_item_type(item)) for item in plan]


def test_simple_fan_graph(simple_fan_graph):
    compiled = simple_fan_graph.compile()
    expected_execution_plan = [
        "process_data",
        "validate",
        ["escape", ["aa", "bb"]],
        "prep",
    ]
    assert create_compable_list(compiled.execution_path) == create_compable_list(
        expected_execution_plan
    )


def test_complex_fan_graph(complex_fan_graph):
    compiled = complex_fan_graph.compile()
    expected_execution_plan = [
        "process_data",
        "validate",
        [["escape", ["cc", "dd"], "hh"], ["aa", "bb"]],
        "prep",
    ]
    assert create_compable_list(compiled.execution_path) == create_compable_list(
        expected_execution_plan
    )
