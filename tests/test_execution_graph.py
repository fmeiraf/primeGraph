import asyncio
import random
import time
from typing import Any, Dict, List

import pytest
from pydantic import Field, ValidationError

from primeGraph.buffer.factory import History, Incremental, LastValue
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState
from primeGraph.types import ChainStatus


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


@pytest.mark.asyncio
async def test_parallel_updates(graph_with_buffers):
    # Execute the graph multiple times
    for _ in range(3):
        await graph_with_buffers.execute()

    # Check the state after execution
    state = graph_with_buffers.state

    # Verify that the counter was incremented correctly
    assert state.counter == 3  # Each execution adds 1

    # Verify that the status was updated to "running"
    assert state.status == "running"

    # Verify that metrics were added correctly
    assert len(state.metrics) == 9  # 3 executions * 3 nodes
    
    # Check each metric individually
    expected_metrics = [
    {"accuracy": 0.95},
    {"precision": 0.90},
    {"recall": 0.85},
]
    for metric in state.metrics:
        # Find matching expected metric
        matching_metric = next((expected for expected in expected_metrics 
                            if list(metric.keys())[0] in expected 
                            and abs(metric[list(metric.keys())[0]] - expected[list(metric.keys())[0]]) < 0.001), None)
        assert matching_metric is not None, f"No matching expected metric found for {metric}"


@pytest.mark.asyncio
async def test_pause_before_node_execution():
    graph = Graph()
    class StateForTest(GraphState):
        execution_order: History[str]

    state = StateForTest(execution_order=[])
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

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", END)
    graph.compile()

    # First execution should stop before task2
    await graph.execute()  # Store the result
    assert graph.state.execution_order == ["task1"]
    assert graph.chain_status == ChainStatus.PAUSE

    # Resume execution - make sure to await the result
    await graph.resume()  # Store the result and await it
    assert graph.state.execution_order == ["task1", "task2", "task3"]
    assert graph.chain_status == ChainStatus.DONE  # Add this check to verify completion


@pytest.mark.asyncio
async def test_pause_after_node_execution():
    graph = Graph()
    class StateForTest(GraphState):
        execution_order: History[str]

    state = StateForTest(execution_order=[])
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

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", END)
    graph.compile()

    # First execution should stop after task2
    await graph.execute()
    assert graph.state.execution_order == ["task1", "task2"]

    # Resume execution
    await graph.resume()
    assert graph.state.execution_order == ["task1", "task2", "task3"]

class StateForTestWithHistory(GraphState):
    execution_order: History[str]


@pytest.mark.asyncio
async def test_multiple_pause_resume_cycles():
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
    await graph.execute()
    assert graph.state.execution_order == ["task1", "task2"]

    # Second resume - completes execution
    await graph.resume()
    assert graph.state.execution_order == ["task1", "task2", "task3", "task4"]


@pytest.mark.asyncio
async def test_pause_resume_with_parallel_execution():
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
    await graph.execute()
    assert "task1" in graph.state.execution_order
    assert "task3" in graph.state.execution_order
    assert "task2" not in graph.state.execution_order
    assert "task4" not in graph.state.execution_order

    # Resume should complete the execution
    await graph.resume()
    assert "task2" in graph.state.execution_order
    assert "task4" in graph.state.execution_order


@pytest.mark.asyncio
async def test_pause_resume_with_parallel_execution_after_version():
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
    graph.add_edge("task1", "task3")
    graph.add_edge("task2", "task4")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # First execution should execute task1 and task3, but pause before task2
    await graph.execute()
    assert "task1" in graph.state.execution_order
    assert "task3" in graph.state.execution_order
    assert "task2" in graph.state.execution_order
    assert "task4" not in graph.state.execution_order

    # Resume should complete the execution
    await graph.resume()
    assert "task4" in graph.state.execution_order


class StateForTestWithInitialValues(GraphState):
    execution_order: History[str]
    counter: Incremental[int]


@pytest.mark.asyncio
async def test_initial_state_with_filled_values():
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
    await graph.execute()
    expected_tasks = {"pre_task", "task0", "task1", "task2", "task3", "task4"}
    assert set(graph.state.execution_order) == expected_tasks
    assert graph.state.counter == 9  # 2 + 3 + 4


@pytest.mark.asyncio
async def test_state_modification_during_execution():
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
    await graph.execute()
    assert "task1" in graph.state.execution_order
    assert "task2" in graph.state.execution_order
    assert "task3" not in graph.state.execution_order
    assert "task4" not in graph.state.execution_order

    graph.update_state_and_checkpoint({"execution_order": "appended_value"})
    assert graph.state.execution_order == ["task1", "task2", "appended_value"]

    # Resume should complete the execution
    await graph.resume()
    assert graph.state.execution_order == [
        "task1",
        "task2",
        "appended_value",
        "task3",
        "task4",
    ]


@pytest.mark.asyncio
async def test_graph_state_simple_types():
    class SimpleState(GraphState):
        counter: Incremental[int]
        status: LastValue[str]
        metrics: History[float]

    # Test valid initialization
    state = SimpleState(counter=0, status="ready", metrics=[1.0, 2.0, 3.0])
    assert state.counter == 0
    assert state.status == "ready"
    assert state.metrics == [1.0, 2.0, 3.0]

    # Test invalid types
    with pytest.raises(TypeError):
        SimpleState(counter="invalid", status="ready", metrics=[1.0])
    with pytest.raises(TypeError):
        SimpleState(counter=0, status=123, metrics=[1.0])
    with pytest.raises(TypeError):
        SimpleState(counter=0, status="ready", metrics=1.0)  # Should be list


@pytest.mark.asyncio
async def test_graph_state_dict_types():
    class DictState(GraphState):
        simple_dict: LastValue[Dict[str, int]]
        nested_dict: LastValue[Dict[str, Dict[str, float]]]
        dict_history: History[Dict[str, str]]

    # Test valid initialization
    state = DictState(
        simple_dict={"a": 1, "b": 2},
        nested_dict={"x": {"y": 1.0}},
        dict_history=[{"status": "start"}, {"status": "end"}]
    )
    assert state.simple_dict == {"a": 1, "b": 2}
    assert state.nested_dict == {"x": {"y": 1.0}}
    assert state.dict_history == [{"status": "start"}, {"status": "end"}]

    # Test invalid types
    with pytest.raises((TypeError, ValidationError)):
        DictState(
            simple_dict=[1, 2],  # Should be dict
            nested_dict={"x": {"y": 1.0}},
            dict_history=[{"status": "start"}]
        )
    with pytest.raises((TypeError, ValidationError)):
        DictState(
            simple_dict={"a": 1},
            nested_dict={"x": 1.0},  # Should be nested dict
            dict_history=[{"status": "start"}]
        )
    with pytest.raises((TypeError, ValidationError)):
        DictState(
            simple_dict={"a": 1},
            nested_dict={"x": {"y": 1.0}},
            dict_history={"status": "start"}  # Should be list
        )


@pytest.mark.asyncio
async def test_graph_state_list_types():
    class ListState(GraphState):
        simple_list: LastValue[List[int]]
        nested_list: LastValue[List[List[str]]]
        list_history: History[List[float]]

    # Test valid initialization
    state = ListState(
        simple_list=[1, 2, 3],
        nested_list=[["a", "b"], ["c", "d"]],
        list_history=[[1.0, 2.0], [3.0, 4.0]]
    )
    assert state.simple_list == [1, 2, 3]
    assert state.nested_list == [["a", "b"], ["c", "d"]]
    assert state.list_history == [[1.0, 2.0], [3.0, 4.0]]

    # Test invalid types
    with pytest.raises((TypeError, ValidationError)):
        ListState(
            simple_list=1,  # Should be list
            nested_list=[["a", "b"]],
            list_history=[[1.0, 2.0]]
        )
    with pytest.raises((TypeError, ValidationError)):
        ListState(
            simple_list=[1, 2],
            nested_list=["a", "b"],  # Should be nested list
            list_history=[[1.0, 2.0]]
        )
    with pytest.raises((TypeError, ValidationError)):
        ListState(
            simple_list=[1, 2],
            nested_list=[["a", "b"]],
            list_history=[1.0, 2.0]  # Should be list of lists
        )


@pytest.mark.asyncio
async def test_graph_state_complex_types():
    class ComplexState(GraphState):
        dict_list: LastValue[Dict[str, List[int]]]
        list_dict: LastValue[List[Dict[str, float]]]
        complex_history: History[Dict[str, List[Dict[str, Any]]]]

    # Test valid initialization
    state = ComplexState(
        dict_list={"a": [1, 2], "b": [3, 4]},
        list_dict=[{"x": 1.0}, {"y": 2.0}],
        complex_history=[
            {"data": [{"value": 1}, {"value": 2}]},
            {"data": [{"value": 3}, {"value": 4}]}
        ]
    )
    assert state.dict_list == {"a": [1, 2], "b": [3, 4]}
    assert state.list_dict == [{"x": 1.0}, {"y": 2.0}]
    assert state.complex_history == [
        {"data": [{"value": 1}, {"value": 2}]},
        {"data": [{"value": 3}, {"value": 4}]}
    ]

    # Test invalid types
    with pytest.raises((TypeError, ValidationError)):
        ComplexState(
            dict_list=[1, 2],  # Should be dict
            list_dict=[{"x": 1.0}],
            complex_history=[{"data": [{"value": 1}]}]
        )
    with pytest.raises((TypeError, ValidationError)):
        ComplexState(
            dict_list={"a": [1, 2]},
            list_dict={"x": 1.0},  # Should be list
            complex_history=[{"data": [{"value": 1}]}]
        )
    with pytest.raises((TypeError, ValidationError)):
        ComplexState(
            dict_list={"a": [1, 2]},
            list_dict=[{"x": 1.0}],
            complex_history={"data": [{"value": 1}]}  # Should be list
        )


@pytest.mark.asyncio
async def test_graph_state_incremental_types():
    class IncrementalState(GraphState):
        simple_counter: Incremental[int]
        float_counter: Incremental[float]
        dict_counter: Incremental[Dict[str, int]]

    # Test valid initialization
    state = IncrementalState(
        simple_counter=0,
        float_counter=0.0,
        dict_counter={"count": 0}
    )
    assert state.simple_counter == 0
    assert state.float_counter == 0.0
    assert state.dict_counter == {"count": 0}

    # Test invalid types
    with pytest.raises((TypeError, ValidationError)):
        IncrementalState(
            simple_counter="0",  # Should be int
            float_counter=0.0,
            dict_counter={"count": 0}
        )
    with pytest.raises((TypeError, ValidationError)):
        IncrementalState(
            simple_counter=0,
            float_counter="0.0",  # Should be float
            dict_counter={"count": 0}
        )
    with pytest.raises((TypeError, ValidationError)):
        IncrementalState(
            simple_counter=0,
            float_counter=0.0,
            dict_counter=[0]  # Should be dict
        )


@pytest.mark.asyncio
async def test_execution_steps_with_interrupt():
    class StateWithSteps(GraphState):
        number_of_executed_steps: Incremental[int]
        current_status: LastValue[str]

    # Initialize state
    state = StateWithSteps(
        number_of_executed_steps=0,
        current_status="initializing"
    )
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {
            "number_of_executed_steps": 1,
            "current_status": "task1_complete"
        }

    @graph.node(interrupt="before")
    def task2(state):
        return {
            "number_of_executed_steps": 1,
            "current_status": "task2_complete"
        }

    @graph.node()
    def task3(state):
        return {
            "number_of_executed_steps": 1,
            "current_status": "task3_complete"
        }

    # Create workflow
    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", END)
    graph.compile()

    # First execution - should stop before task2
    await graph.execute()
    assert graph.state.number_of_executed_steps == 1  # Only task1 executed
    assert graph.state.current_status == "task1_complete"

    # Resume execution - should complete remaining tasks
    await graph.resume()
    assert graph.state.number_of_executed_steps == 3  # All tasks executed
    assert graph.state.current_status == "task3_complete"


@pytest.mark.asyncio
async def test_state_update_with_buffers():
    class TestState(GraphState):
        counter: Incremental[int]
        status: LastValue[str]
        metrics: History[Dict[str, float]]

    # Initialize state with some values
    initial_state = TestState(
        counter=5,
        status="initial",
        metrics=[{"accuracy": 0.9}]
    )
    graph = Graph(state=initial_state)

    # Test partial update (key-only)
    graph.update_state_and_checkpoint({"counter": 3})
    # Buffer should be consumed when updating state
    assert graph.state.counter == 8  # 5 + 3 (Incremental)
    assert graph.state.status == "initial"  # Unchanged
    assert graph.state.metrics == [{"accuracy": 0.9}]  # Unchanged
    assert not graph.buffers['counter']._ready_for_consumption  # Buffer should be consumed
    
    # Test multiple keys update
    graph.update_state_and_checkpoint({
        "status": "running",
        "metrics": {"precision": 0.85}
    })
    assert graph.state.counter == 8  # Unchanged
    assert graph.state.status == "running"  # Updated (LastValue)
    assert graph.state.metrics == [{"accuracy": 0.9}, {"precision": 0.85}]  # Appended (History)
    assert not graph.buffers['status']._ready_for_consumption  # Buffer should be consumed
    assert not graph.buffers['metrics']._ready_for_consumption  # Buffer should be consumed

    # Verify buffer values are in sync but consumed
    for field_name, buffer in graph.buffers.items():
        assert not buffer._ready_for_consumption  # All buffers should be consumed
        assert getattr(graph.state, field_name) == buffer.value  # But values should match


@pytest.mark.asyncio
async def test_state_set_complete_reset():
    class TestState(GraphState):
        counter: Incremental[int]
        status: LastValue[str]
        metrics: History[Dict[str, float]]

    # Initialize state with some values
    initial_state = TestState(
        counter=5,
        status="initial",
        metrics=[{"accuracy": 0.9}]
    )
    graph = Graph(state=initial_state)

    # Test complete state reset
    new_state = TestState(
        counter=10,
        status="complete",
        metrics=[{"final": 0.95}]
    )
    graph.set_state_and_checkpoint(new_state)
    assert graph.state.counter == 10  # Complete reset, not incremental
    assert graph.state.status == "complete"  # New value
    assert graph.state.metrics == [{"final": 0.95}]  # New list, not appended

    # Test partial state update via dict (should still reset those fields)
    graph.set_state_and_checkpoint({
        "counter": 3,
        "metrics": [{"new": 0.80}]
    })
    assert graph.state.counter == 3  # Reset to 3, not added to 10
    assert graph.state.status == "complete"  # Unchanged
    assert graph.state.metrics == [{"new": 0.80}]  # Reset to new list, not appended


@pytest.mark.asyncio
async def test_state_update_validation():
    class TestState(GraphState):
        counter: Incremental[int]
        status: LastValue[str]
        metrics: History[Dict[str, float]]

    initial_state = TestState(
        counter=0,
        status="initial",
        metrics=[{"accuracy": 0.9}]
    )
    graph = Graph(state=initial_state)

    # Test invalid key for both methods
    with pytest.raises(ValueError, match="Invalid state fields"):
        graph.update_state_and_checkpoint({"invalid_key": 123})
    with pytest.raises(ValueError, match="Invalid state fields"):
        graph.set_state_and_checkpoint({"invalid_key": 123})

    # Test invalid types
    with pytest.raises((TypeError, ValidationError)):
        graph.update_state_and_checkpoint({"counter": "not_an_int"})
    with pytest.raises((TypeError, ValidationError)):
        graph.set_state_and_checkpoint({"counter": "not_an_int"})

    # Test invalid model type
    class DifferentState(GraphState):
        field: LastValue[str]

    with pytest.raises(ValueError, match="must be an instance of"):
        graph.set_state_and_checkpoint(DifferentState(field="test"))

    # Verify state remained unchanged after failed updates
    assert graph.state.counter == 0
    assert graph.state.status == "initial"
    assert graph.state.metrics == [{"accuracy": 0.9}]


@pytest.mark.asyncio
async def test_buffer_behavior_differences():
    class BufferTestState(GraphState):
        last_value: LastValue[str]
        history: History[str]
        increment: Incremental[int]

    initial_state = BufferTestState(
        last_value="initial",
        history=["first"],
        increment=0
    )
    graph = Graph(state=initial_state)

    # Test update behavior (using buffers)
    graph.update_state_and_checkpoint({
        "last_value": "update1",
        "history": "second",
        "increment": 5
    })
    assert graph.state.last_value == "update1"  # LastValue: replaced
    assert graph.state.history == ["first", "second"]  # History: appended
    assert graph.state.increment == 5  # Incremental: added to 0

    graph.update_state_and_checkpoint({
        "last_value": "update2",
        "history": "third",
        "increment": 3
    })
    assert graph.state.last_value == "update2"  # LastValue: replaced
    assert graph.state.history == ["first", "second", "third"]  # History: appended
    assert graph.state.increment == 8  # Incremental: added to 5

    # Test set behavior (direct replacement)
    graph.set_state_and_checkpoint({
        "last_value": "set1",
        "history": ["new_first"],
        "increment": 10
    })
    assert graph.state.last_value == "set1"  # Direct replacement
    assert graph.state.history == ["new_first"]  # Complete reset of list
    assert graph.state.increment == 10  # Direct replacement, not incremental

    # Verify that subsequent updates after set follow buffer rules
    graph.update_state_and_checkpoint({
        "last_value": "update3",
        "history": "new_second",
        "increment": 5
    })
    assert graph.state.last_value == "update3"  # LastValue: replaced
    assert graph.state.history == ["new_first", "new_second"]  # History: appended to reset list
    assert graph.state.increment == 15  # Incremental: added to reset value


@pytest.mark.asyncio
async def test_chain_status_after_completion():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1"}

    @graph.node()
    def task2(state):
        return {"execution_order": "task2"}

    # Create simple sequential path
    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", END)
    graph.compile()

    # Execute the graph
    await graph.execute()

    # Verify execution completed and chain status is DONE
    assert graph.state.execution_order == ["task1", "task2"]
    assert graph.chain_status == ChainStatus.DONE


@pytest.mark.asyncio
async def test_chain_status_with_interrupts():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="before")
    def task2(state):
        return {"execution_order": "task2"}

    @graph.node(interrupt="after")
    def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    def task4(state):
        return {"execution_order": "task4"}

    # Create path with interrupts
    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # First execution - should stop before task2
    await graph.execute()
    assert graph.chain_status == ChainStatus.PAUSE
    assert graph.state.execution_order == ["task1"]

    # Second execution - should stop after task3
    await graph.resume()
    assert graph.chain_status == ChainStatus.PAUSE
    assert graph.state.execution_order == ["task1", "task2", "task3"]

    # Final execution - should complete and set status to DONE
    await graph.resume()
    assert graph.chain_status == ChainStatus.DONE
    assert graph.state.execution_order == ["task1", "task2", "task3", "task4"]


@pytest.mark.asyncio
async def test_async_parallel_execution_timeout():
    basic_graph = Graph()

    @basic_graph.node()
    async def slow_task():
        await asyncio.sleep(10)  # Task that takes too long

    @basic_graph.node()
    async def normal_task():
        pass

    basic_graph.add_edge(START, "slow_task")
    basic_graph.add_edge("slow_task", "normal_task")
    basic_graph.add_edge("normal_task", END)
    basic_graph.compile()

    # Verify timeout error is raised
    with pytest.raises(TimeoutError) as exc_info:
        await basic_graph.execute(timeout=2)

    assert "Execution timeout" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_parallel_execution_with_error():
    basic_graph = Graph()

    @basic_graph.node()
    async def failing_task():
        raise ValueError("Task failed")

    @basic_graph.node()
    async def normal_task():
        pass

    basic_graph.add_edge(START, "failing_task")
    basic_graph.add_edge("failing_task", "normal_task")
    basic_graph.add_edge("normal_task", END)
    basic_graph.compile()

    # Verify the error is propagated
    with pytest.raises(RuntimeError) as exc_info:
        await basic_graph.execute()

    assert "Task failed" in str(exc_info.value)


# TODO: This test is not completely correct. Interrupting a branch while the other execute until convergence is not implemented yet.
@pytest.mark.asyncio
async def test_parallel_execution_with_interrupt():
    class StateForTestWithHistory(GraphState):
        execution_order: History[str]
        counter: Incremental[int]

    state = StateForTestWithHistory(execution_order=[], counter=0)
    graph = Graph(state=state)

    @graph.node()
    def task1(state):
        return {"execution_order": "task1", "counter": 1}

    @graph.node(interrupt="after")
    def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    def task4(state):
        return {"execution_order": "task4"}
    
    @graph.node()
    def task5(state):
        return {"execution_order": "task5"}
    
    @graph.node()
    def task6(state):
        return {"execution_order": "task6"}

    # Create parallel paths
    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task1", "task3")
    graph.add_edge("task2", "task4")
    graph.add_edge("task3", "task5")
    graph.add_edge("task4", "task6")
    graph.add_edge("task5", "task6")
    graph.add_edge("task6", END)
    graph.compile()

    # First execution - should execute task1, task2, task3, and task5
    # but pause after task2 before executing task4
    await graph.execute()
    assert set(graph.state.execution_order) == {"task1", "task2", "task3"} #, "task5"}
    assert graph.chain_status == ChainStatus.PAUSE
    assert graph.state.counter == 1

    # Resume execution - should complete task4 and task6
    await graph.resume()
    assert set(graph.state.execution_order) == {"task1", "task2", "task3", "task4", "task5", "task6"}
    assert graph.chain_status == ChainStatus.DONE
    
    # Verify execution order: task4 must come before task6
    task4_index = graph.state.execution_order.index("task4")
    task6_index = graph.state.execution_order.index("task6")
    assert task4_index < task6_index, "task4 should be executed before task6"


class SimpleTestState(GraphState):
    # This state keeps track of the order of node execution and the random decision outcome.
    execution_order: History[str] = Field(default_factory=list)
    user_message: LastValue[str] = ""
    is_followup: LastValue[bool] = False
    is_summarize: LastValue[bool] = False
    is_finalize: LastValue[bool] = False


def create_simple_graph() -> Graph:
    state = SimpleTestState()
    graph = Graph(state=state)

    @graph.node()
    def start_conversation(state: SimpleTestState):
        print("Executing start_conversation")
        state.execution_order.append("start_conversation")
        time.sleep(0.1)
        # Simulate asking the user for input by setting a default message.
        return {"user_message": "I want to plan something"}

    @graph.node()
    def process_user_message(state: SimpleTestState):
        print("Executing process_user_message")
        state.execution_order.append("process_user_message")
        time.sleep(0.1)
        # Random decision among three outcomes
        outcome = random.choice(["followup", "summarize", "finalize"])
        print(f"Random outcome selected: {outcome}")
        if outcome == "followup":
            return {"is_followup": True, "is_summarize": False, "is_finalize": False}
        elif outcome == "summarize":
            return {"is_followup": False, "is_summarize": True, "is_finalize": False}
        else:
            return {"is_followup": False, "is_summarize": False, "is_finalize": True}

    @graph.node()
    def response_router(state: SimpleTestState) -> str:
        print("Executing response_router")
        state.execution_order.append("response_router")
        time.sleep(0.1)
        # Route according to the flags set by process_user_message:
        if state.is_finalize:
            return "finalize"
        elif state.is_summarize:
            return "summarize"
        elif state.is_followup:
            return "followup"
        return END

    @graph.node()
    def followup(state: SimpleTestState):
        print("Executing followup branch")
        state.execution_order.append("followup")
        time.sleep(0.1)
        return {}

    @graph.node()
    def summarize(state: SimpleTestState):
        print("Executing summarize branch")
        state.execution_order.append("summarize")
        time.sleep(0.1)
        return {}

    @graph.node()
    def finalize(state: SimpleTestState):
        print("Executing finalize branch")
        state.execution_order.append("finalize")
        time.sleep(0.1)
        return {}

    # Build the graph:
    graph.add_edge(START, "start_conversation")
    graph.add_edge("start_conversation", "process_user_message")
    graph.add_router_edge("process_user_message", "response_router")
    graph.add_edge("response_router", "followup")
    graph.add_edge("response_router", "summarize")
    graph.add_edge("response_router", "finalize")
    graph.add_edge("followup", END)
    graph.add_edge("summarize", END)
    graph.add_edge("finalize", END)

    graph.compile()
    return graph


@pytest.mark.asyncio
async def test_simple_random_graph():
    graph = create_simple_graph()
    await graph.execute()
    # In case the graph paused (for example due to an interrupt), resume execution.
    if graph.chain_status != "DONE":
        await graph.resume()
    print("Final execution order:", graph.state.execution_order)
    # Expected order should be: start_conversation, process_user_message, response_router, then exactly one branch node.
    assert "start_conversation" in graph.state.execution_order
    assert "process_user_message" in graph.state.execution_order
    assert "response_router" in graph.state.execution_order
    branch_nodes = {"followup", "summarize", "finalize"}
    executed_branches = set(graph.state.execution_order).intersection(branch_nodes)
    assert len(executed_branches) == 1, "Exactly one branch should have been executed"


@pytest.mark.asyncio
async def test_any_type_in_state():
    """Test that using typing.Any in state type annotations works correctly."""
    from typing import Any, Dict, List
    
    class StateWithAnyType(GraphState):
        any_value: LastValue[Any]  # Using Any directly
        any_dict: LastValue[Dict[str, Any]]  # Any in dict values
        any_history: History[Any]  # Any in History
        any_list_history: History[List[Any]]  # List of Any in History

    # Initialize with various types
    state = StateWithAnyType(
        any_value="string value",
        any_dict={"key1": 123, "key2": "value", "key3": {"nested": True}},
        any_history=["item1", 2, {"key": "value"}],
        any_list_history=[[1, 2, 3], ["a", "b", "c"]]
    )
    
    graph = Graph(state=state)

    @graph.node()
    def update_any_values(state):
        return {
            "any_value": 12345,  # Change type from string to int
            "any_dict": {"new_key": [1, 2, 3]},  # Change value types
            "any_history": {"completely": "different", "type": True},  # Add dict to history
            "any_list_history": [{"mixed": "types", "in": 1, "list": True}]  # Add mixed types
        }

    @graph.node()
    def update_again(state):
        return {
            "any_value": {"now": "a dict"},  # Change type again
            "any_dict": {"key": None},  # Add None value
            "any_history": None,  # Add None to history
            "any_list_history": [None]  # Add None to list history
        }

    @graph.node()
    def complex_review_and_edit(state):
        """This tests the node name that was causing an error in your real code."""
        # Return an object with complex nested structure
        return {
            "any_dict": {
                "deeply": {
                    "nested": [1, "mixed", {"types": True}]
                }
            }
        }

    graph.add_edge(START, "update_any_values")
    graph.add_edge("update_any_values", "update_again")
    graph.add_edge("update_again", "complex_review_and_edit")
    graph.add_edge("complex_review_and_edit", END)
    graph.compile()

    # Execute the graph
    await graph.execute()

    # Verify values were updated correctly with different types
    assert isinstance(graph.state.any_value, dict)
    assert graph.state.any_value == {"now": "a dict"}
    
    assert isinstance(graph.state.any_dict, dict)
    assert graph.state.any_dict == {"deeply": {"nested": [1, "mixed", {"types": True}]}}
    
    # For History buffer, we should see all values in the history
    assert len(graph.state.any_history) == 5
    assert graph.state.any_history[0] == "item1"
    assert graph.state.any_history[3] == {"completely": "different", "type": True}
    assert graph.state.any_history[4] is None
    
    assert len(graph.state.any_list_history) == 4
    assert graph.state.any_list_history[0] == [1, 2, 3]
    assert graph.state.any_list_history[1] == ["a", "b", "c"]
    assert graph.state.any_list_history[2] == [{"mixed": "types", "in": 1, "list": True}]
    assert graph.state.any_list_history[3] == [None]