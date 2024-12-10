import time
from typing import Dict

import pytest

from tiny_graph.buffer.factory import History, Incremental, LastValue
from tiny_graph.constants import END, START
from tiny_graph.graph.executable import ExecutableNode, Graph
from tiny_graph.models.state import GraphState


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
    state = ComplexTestState(counter=0, status="", metrics=[])
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
    basic_graph.detailed_execution_path = [
        ("__start__", "process_data"),
        ("process_data", "validate"),
    ]
    result = basic_graph._convert_execution_plan()

    assert len(result) == 2
    assert all(isinstance(node, ExecutableNode) for node in result)
    assert result[0].node_name == "process_data"
    assert result[0].execution_type == "sequential"
    assert len(result[0].task_list) == 1

    # Test parallel execution
    basic_graph.detailed_execution_path = [
        [("validate", "aa"), ("bb", "bb")],
        [
            ("escape", "escape"),
            [("escape", "dd"), ("escape", "cc")],
            ("validate", "hh"),
        ],
    ]
    result = basic_graph._convert_execution_plan()

    # Execpted result:
    #     [
    #     ExecutableNode(
    #         node_name='group_aa_bb',
    #         task_list=[<function aa at 0x10c756ac0>, <function bb at 0x10c7551c0>],
    #         node_list=['aa', 'bb'],
    #         execution_type='sequential',
    #         interrupt=None
    #     ),
    #     ExecutableNode(
    #         node_name='group_escape_(dd_cc)_hh',
    #         task_list=[
    #             <function escape at 0x10c756b60>,
    #             ExecutableNode(
    #                 node_name='group_dd_cc',
    #                 task_list=[<function dd at 0x10c7568e0>, <function cc at 0x10c755ee0>],
    #                 node_list=['dd', 'cc'],
    #                 execution_type='parallel',
    #                 interrupt=None
    #             ),
    #             <function hh at 0x10c756c00>
    #         ],
    #         node_list=['escape', ['dd', 'cc'], 'hh'],
    #         execution_type='sequential',
    #         interrupt=None
    #     )
    # ]

    assert len(result) == 2
    assert result[1].task_list[1].node_name == "group_dd_cc"
    assert result[1].task_list[1].execution_type == "parallel"
    assert len(result[1].task_list) == 3


def test_execution_plan_invalid_input(basic_graph):
    # Test invalid input
    basic_graph.detailed_execution_path = [None]
    with pytest.raises(ValueError):
        basic_graph._convert_execution_plan()


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
    basic_graph.start()

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
        basic_graph.start()

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
        basic_graph.start(timeout=1)

    assert "Execution timeout" in str(exc_info.value)


class StateForTest(GraphState):
    counter: Incremental[int]
    status: LastValue[str]
    metrics: History[dict]


@pytest.fixture
def graph_with_buffers():
    state = StateForTest(counter=0, status="", metrics=[])
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
        graph_with_buffers.start()

    # Check the state after execution
    state = graph_with_buffers.state

    # Verify that the counter was incremented 3 times
    assert state.counter == 1

    # Verify that the status was updated to "running"
    assert state.status == "running"

    # Verify that metrics were added 9 times (3 executions * 3 nodes)
    assert len(state.metrics) == 3
    expected_metrics = [
        {"accuracy": 0.95},
        {"precision": 0.90},
        {"recall": 0.85},
    ]
    for metric in state.metrics:
        assert metric in expected_metrics


def test_pause_before_node_execution():
    graph = Graph()
    execution_order = []

    @graph.node()
    def task1():
        execution_order.append("task1")

    @graph.node(interrupt="before")
    def task2():
        execution_order.append("task2")

    @graph.node()
    def task3():
        execution_order.append("task3")

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", END)
    graph.compile()

    # First execution should stop before task2
    graph.start()
    assert execution_order == ["task1"]
    assert graph.next_execution_node == "task2"

    # Resume execution
    graph.resume()
    assert execution_order == ["task1", "task2", "task3"]


def test_pause_after_node_execution():
    graph = Graph()
    execution_order = []

    @graph.node()
    def task1():
        execution_order.append("task1")

    @graph.node(interrupt="after")
    def task2():
        execution_order.append("task2")

    @graph.node()
    def task3():
        execution_order.append("task3")

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", END)
    graph.compile()

    # First execution should stop after task2
    graph.start()
    assert execution_order == ["task1", "task2"]
    assert graph.next_execution_node == "task3"

    # Resume execution
    graph.resume()
    assert execution_order == ["task1", "task2", "task3"]


def test_resume_without_pause():
    graph = Graph()

    @graph.node()
    def task1():
        pass

    @graph.node()
    def task2():
        pass

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", END)
    graph.compile()

    # Should raise error when trying to resume without a pause
    with pytest.raises(ValueError):
        graph.resume()


class StateForTestWithHistory(GraphState):
    execution_order: History[str]


def test_multiple_pause_resume_cycles():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="after")
    def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    def task4(state):
        return {"execution_order": "task4"}

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # First execution - stops after task2
    graph.start()
    assert graph.state.execution_order == ["task1", "task2"]
    assert graph.next_execution_node == "task3"

    # Second resume - completes execution
    graph.resume()
    assert graph.state.execution_order == ["task1", "task2", "task3", "task4"]


def test_pause_resume_with_parallel_execution():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="before")
    def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    def task4(state):
        return {"execution_order": "task4"}

    # Create parallel paths
    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task1", "task3")
    graph.add_edge("task2", "task4")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # First execution should execute task1 and task3, but pause before task2
    graph.start()
    assert "task1" in graph.state.execution_order
    assert "task3" in graph.state.execution_order
    assert "task2" not in graph.state.execution_order
    assert "task4" not in graph.state.execution_order
    assert graph.next_execution_node == "task2"

    # Resume should complete the execution
    graph.resume()
    assert "task2" in graph.state.execution_order
    assert "task4" in graph.state.execution_order


def test_resume_with_start_from_only():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1"}

    @graph.node()
    def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    def task4(state):
        return {"execution_order": "task4"}

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # Start execution from task2
    graph.resume(start_from="task2")
    assert graph.state.execution_order == ["task2", "task3", "task4"]


class StateForTestWithInitialValues(GraphState):
    execution_order: History[str]
    counter: Incremental[int]


def test_initial_state_with_filled_values():
    state = StateForTestWithInitialValues(
        execution_order=["pre_task", "task0"], counter=2
    )
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1", "counter": 3}

    @graph.node()
    def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    def task3(state):
        return {"execution_order": "task3", "counter": 4}

    @graph.node()
    def task4(state):
        return {"execution_order": "task4"}

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # Start execution from task2
    graph.start()
    expected_tasks = {"pre_task", "task0", "task1", "task2", "task3", "task4"}
    assert set(graph.state.execution_order) == expected_tasks
    assert graph.state.counter == 9  # 2 + 3 + 4


def test_state_modification_during_execution():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="after")
    def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    def task4(state):
        return {"execution_order": "task4"}

    # Create parallel paths
    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # First execution should execute task1 and task3, but pause before task2
    graph.start()
    assert "task1" in graph.state.execution_order
    assert "task2" in graph.state.execution_order
    assert "task3" not in graph.state.execution_order
    assert "task4" not in graph.state.execution_order

    state.execution_order.append("appended_value")
    assert state.execution_order == ["task1", "task2", "appended_value"]

    # Resume should complete the execution
    graph.resume()
    assert graph.state.execution_order == [
        "task1",
        "task2",
        "appended_value",
        "task3",
        "task4",
    ]
