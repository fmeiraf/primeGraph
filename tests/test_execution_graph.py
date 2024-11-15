import time
from typing import Dict

import pytest

from tiny_graph.buffer.factory import History, Incremental, LastValue
from tiny_graph.constants import END, START
from tiny_graph.graph.executable import ExecutableNode, Graph
from tiny_graph.models.base import GraphState


@pytest.fixture
def basic_graph():
    simple_graph = Graph()

    # Define some example actions
    @simple_graph.node()
    def escape():
        print("Starting workflow")

    @simple_graph.node()
    def process_data():
        print("Processing data")

    @simple_graph.node()
    def validate():
        print("Validating results")

    @simple_graph.node()
    def aa():
        print("Validating results")

    @simple_graph.node()
    def bb():
        print("Validating results")

    @simple_graph.node()
    def dd():
        print("Validating results")

    @simple_graph.node()
    def cc():
        print("Validating results")

    @simple_graph.node()
    def hh():
        print("Validating results")

    @simple_graph.node()
    def prep():
        print("Workflow complete")

    # Add edges to create workflow
    simple_graph.add_edge(START, "process_data")
    simple_graph.add_edge("process_data", "validate")
    simple_graph.add_edge("validate", "escape")
    simple_graph.add_edge("escape", "dd")
    simple_graph.add_edge("escape", "cc")
    simple_graph.add_edge("cc", "hh")
    simple_graph.add_edge("dd", "hh")
    simple_graph.add_edge("hh", "prep")
    simple_graph.add_edge("validate", "aa")
    simple_graph.add_edge("aa", "bb")
    simple_graph.add_edge("bb", "prep")
    simple_graph.add_edge("prep", END)

    simple_graph.compile()

    return simple_graph


@pytest.fixture
def complex_graph():
    class ComplexTestState(GraphState):
        counter: Incremental[int]  # Will accumulate values
        status: LastValue[str]  # Will only keep last value
        metrics: History[Dict[str, float]]  # Will keep history of all updates

    # Initialize the graph with state
    state = ComplexTestState(counter=0, status="", metrics={})
    graph = Graph(state=state)

    # Define nodes (same as in your notebook)
    @graph.node()
    def increment_counter(state):
        return {"counter": 2}

    @graph.node()
    def decrement_counter(state):
        return {"counter": -1}

    @graph.node()
    def update_status_to_in_progress(state):
        return {"status": "in_progress"}

    @graph.node()
    def update_status_to_complete(state):
        return {"status": "complete"}

    @graph.node()
    def add_metrics(state):
        return {"metrics": {"accuracy": 0.9, "loss": 0.1}}

    @graph.node()
    def update_metrics(state):
        return {"metrics": {"loss": 0.05, "precision": 0.85}}

    @graph.node()
    def finalize_metrics(state):
        return {"metrics": {"finalized": True}}

    # Create the workflow with multiple levels of execution
    graph.add_edge(START, "increment_counter")
    graph.add_edge(START, "decrement_counter")
    graph.add_edge(START, "update_status_to_in_progress")
    graph.add_edge("increment_counter", "add_metrics")
    graph.add_edge("decrement_counter", "add_metrics")
    graph.add_edge("add_metrics", "update_metrics")
    graph.add_edge("update_metrics", "finalize_metrics")
    graph.add_edge("update_status_to_in_progress", "update_status_to_complete")
    graph.add_edge("update_status_to_complete", "finalize_metrics")
    graph.add_edge("finalize_metrics", END)

    graph.compile()

    return graph


def extract_executable_nodes_info(executable_node):
    if len(executable_node.task_list) <= 1:
        return (executable_node.task_list[0], executable_node.execution_type)
    else:
        return [
            extract_executable_nodes_info(task) for task in executable_node.task_list
        ]


def test_execution_plan_conversion(basic_graph):
    # Test sequential execution
    basic_graph.execution_plan = ["process_data", "validate"]
    result = basic_graph._convert_execution_plan()

    assert len(result) == 2
    assert all(isinstance(node, ExecutableNode) for node in result)
    assert result[0].node_name == "process_data"
    assert result[0].execution_type == "sequential"
    assert len(result[0].task_list) == 1

    # Test parallel execution
    basic_graph.execution_plan = [["escape", "aa"]]
    result = basic_graph._convert_execution_plan()

    assert len(result) == 1
    assert result[0].node_name == "group_escape"
    assert result[0].execution_type == "parallel"
    assert len(result[0].task_list) == 2

    # Test mixed execution
    basic_graph.execution_plan = ["process_data", ["escape", "cc"], "prep"]
    result = basic_graph._convert_execution_plan()

    assert len(result) == 3
    assert result[0].execution_type == "sequential"
    assert result[1].execution_type == "parallel"
    assert result[2].execution_type == "sequential"
    assert len(result[1].task_list) == 2


def test_execution_plan_invalid_input(basic_graph):
    # Test invalid input
    basic_graph.execution_plan = [None]
    with pytest.raises(ValueError):
        basic_graph._convert_execution_plan()


def test_find_execution_paths_with_edges(basic_graph):
    # Get the execution plan with edges
    edge_plan = basic_graph._find_execution_paths_with_edges()

    def flatten_edge_plan(plan):
        """Helper to flatten the nested structure into a set of edge IDs"""
        edges = set()
        for item in plan:
            if isinstance(item, str):
                edges.add(item)
            elif isinstance(item, list):
                edges.update(flatten_edge_plan(item))
        return edges

    # Get all edges from the plan
    all_edges = flatten_edge_plan(edge_plan)

    # Verify required edges exist
    expected_edges = {
        "__start___to_process_data_1",
        "process_data_to_validate_1",
        "validate_to_escape_1",
        "validate_to_aa_1",
        "escape_to_dd_1",
        "escape_to_cc_1",
        "dd_to_hh_1",
        "cc_to_hh_1",
        "hh_to_prep_1",
        "aa_to_bb_1",
        "bb_to_prep_1",
    }
    assert expected_edges == all_edges

    def verify_structure(plan):
        """Verify the structural properties of the execution plan"""
        if not plan:
            return True

        # If it's a single edge, it should be a string
        if isinstance(plan, str):
            return True

        # If it's a list, verify each element
        if isinstance(plan, list):
            # Parallel paths should be lists within lists
            for item in plan:
                if not (isinstance(item, str) or isinstance(item, list)):
                    return False
                if not verify_structure(item):
                    return False
            return True

        return False

    # Verify the overall structure is valid
    assert verify_structure(edge_plan)

    # Verify some key relationships without caring about exact order
    def find_path_containing(edges, start_edge, end_edge):
        """Check if there exists a path containing both edges in the nested structure"""

        def check_sublist(sublist):
            if isinstance(sublist, str):
                return False

            found_start = False
            found_end = False

            for item in sublist:
                if isinstance(item, str):
                    if item == start_edge:
                        found_start = True
                    if item == end_edge:
                        found_end = True
                elif isinstance(item, list):
                    if check_sublist(item):
                        return True

            return found_start and found_end

        return check_sublist(edges)

    # Verify some key path relationships
    assert find_path_containing(
        edge_plan, "__start___to_process_data_1", "process_data_to_validate_1"
    )

    # Verify parallel paths exist after validate
    parallel_paths_exist = any(
        isinstance(item, list) and len(item) > 1 for item in edge_plan
    )
    assert parallel_paths_exist


def test_find_execution_paths_with_edges_for_complex_graph(complex_graph):
    # Get the execution plan with edges
    edge_plan = complex_graph._find_execution_paths_with_edges()

    def flatten_edge_plan(plan):
        """Helper to flatten the nested structure into a set of edge IDs"""
        edges = set()
        for item in plan:
            if isinstance(item, str):
                edges.add(item)
            elif isinstance(item, list):
                edges.update(flatten_edge_plan(item))
        return edges

    # Get all edges from the plan
    all_edges = flatten_edge_plan(edge_plan)

    # Verify required edges exist
    expected_edges = {
        "__start___to_increment_counter_1",
        "__start___to_decrement_counter_1",
        "__start___to_update_status_to_in_progress_1",
        "increment_counter_to_add_metrics_1",
        "decrement_counter_to_add_metrics_1",
        "add_metrics_to_update_metrics_1",
        "update_metrics_to_finalize_metrics_1",
        "update_status_to_in_progress_to_update_status_to_complete_1",
        "update_status_to_complete_to_finalize_metrics_1",
    }
    assert expected_edges == all_edges

    def verify_structure(plan):
        """Verify the structural properties of the execution plan"""
        if not plan:
            return True

        # If it's a single edge, it should be a string
        if isinstance(plan, str):
            return True

        # If it's a list, verify each element
        if isinstance(plan, list):
            # Parallel paths should be lists within lists
            for item in plan:
                if not (isinstance(item, str) or isinstance(item, list)):
                    return False
                if not verify_structure(item):
                    return False
            return True

        return False

    # Verify the overall structure is valid
    assert verify_structure(edge_plan)

    # Verify some key relationships without caring about exact order
    def find_path_containing(edges, start_edge, end_edge):
        """Check if there exists a path containing both edges in the nested structure"""

        def check_sublist(sublist):
            if isinstance(sublist, str):
                return False

            found_start = False
            found_end = False

            for item in sublist:
                if isinstance(item, str):
                    if item == start_edge:
                        found_start = True
                    if item == end_edge:
                        found_end = True
                elif isinstance(item, list):
                    if check_sublist(item):
                        return True

            return found_start and found_end

        return check_sublist(edges)

    # Verify some key path relationships
    # assert find_path_containing(
    #     edge_plan,
    #     "__start___to_increment_counter_1",
    #     "increment_counter_to_add_metrics_1",
    # )

    # Verify parallel paths exist after start
    parallel_paths_exist = any(
        isinstance(item, list) and len(item) > 1 for item in edge_plan
    )
    assert parallel_paths_exist


def test_parallel_execution():
    # Create a list to track execution order
    execution_order = []

    basic_graph = Graph()

    # Override the existing nodes with new ones that track execution
    @basic_graph.node()
    def task1():
        execution_order.append("task1")

    @basic_graph.node()
    def task2():
        execution_order.append("task2")

    @basic_graph.node()
    def task3():
        execution_order.append("task3")

    basic_graph.add_edge(START, "task1")
    basic_graph.add_edge("task1", "task2")
    basic_graph.add_edge("task1", "task3")
    basic_graph.add_edge("task2", END)
    basic_graph.add_edge("task3", END)
    basic_graph.compile()

    # Execute the graph
    basic_graph.execute()

    # Verify task1 was executed first
    assert execution_order[0] == "task1"

    # Verify task2 and task3 were both executed after task1
    assert set(execution_order[1:]) == {"task2", "task3"}
    assert len(execution_order) == 3


def test_parallel_execution_with_error():
    basic_graph = Graph()

    @basic_graph.node()
    def failing_task():
        raise ValueError("Task failed")

    @basic_graph.node()
    def normal_task():
        pass

    basic_graph.add_edge(START, "failing_task")
    basic_graph.add_edge("failing_task", "normal_task")
    basic_graph.add_edge("normal_task", END)
    basic_graph.compile()

    # Verify the error is propagated
    with pytest.raises(RuntimeError) as exc_info:
        basic_graph.execute()

    assert "Task failed" in str(exc_info.value)


def test_parallel_execution_timeout():
    basic_graph = Graph()

    @basic_graph.node()
    def slow_task():
        time.sleep(3)  # Task that takes too long

    @basic_graph.node()
    def normal_task():
        pass

    basic_graph.add_edge(START, "slow_task")
    basic_graph.add_edge("slow_task", "normal_task")
    basic_graph.add_edge("normal_task", END)
    basic_graph.compile()

    # Verify timeout error is raised
    with pytest.raises(TimeoutError) as exc_info:
        basic_graph.execute(timeout=1)

    assert "Execution timeout" in str(exc_info.value)


class TestState(GraphState):
    counter: Incremental[int]
    status: LastValue[str]
    metrics: History[dict]


@pytest.fixture
def graph_with_buffers():
    state = TestState(counter=0, status="", metrics={})
    graph = Graph(state=state)

    @graph.node()
    def increment_counter(state):
        return {"counter": 1}

    @graph.node()
    def update_status(state):
        return {"status": "running"}

    @graph.node()
    def add_metrics_1(state):
        return {"metrics": {"accuracy": 0.95}}

    @graph.node()
    def add_metrics_2(state):
        return {"metrics": {"precision": 0.90}}

    @graph.node()
    def add_metrics_3(state):
        return {"metrics": {"recall": 0.85}}

    # Add edges to create parallel execution paths
    graph.add_edge(START, "increment_counter")
    graph.add_edge(START, "update_status")
    graph.add_edge(START, "add_metrics_1")
    graph.add_edge(START, "add_metrics_2")
    graph.add_edge(START, "add_metrics_3")
    graph.add_edge("increment_counter", END)
    graph.add_edge("update_status", END)
    graph.add_edge("add_metrics_1", END)
    graph.add_edge("add_metrics_2", END)
    graph.add_edge("add_metrics_3", END)
    graph.compile()

    return graph


def test_parallel_updates(graph_with_buffers):
    # Execute the graph multiple times
    for _ in range(3):
        graph_with_buffers.execute()

    # Check the state after execution
    state = graph_with_buffers.state

    # Verify that the counter was incremented 3 times
    assert state.counter == 3

    # Verify that the status was updated to "running"
    assert state.status == "running"

    # Verify that metrics were added 9 times (3 executions * 3 nodes)
    assert len(state.metrics) == 9
    expected_metrics = [
        {"accuracy": 0.95},
        {"precision": 0.90},
        {"recall": 0.85},
    ]
    for metric in state.metrics:
        assert metric in expected_metrics
