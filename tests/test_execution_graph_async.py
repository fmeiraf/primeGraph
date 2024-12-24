import asyncio
from typing import Dict

import pytest

from primeGraph.buffer.factory import History, Incremental, LastValue
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState


class StateForTestWithHistory(GraphState):
    execution_order: History[str]


class StateForTestWithHistoryIncremental(GraphState):
    execution_order: History[str]
    counter: Incremental[int]


class StateForTestWithBuffers(GraphState):
    counter: Incremental[int]
    status: LastValue[str]
    metrics: History[Dict[str, float]]


class StateForTestParallelExecution(GraphState):
    execution_order: History[str]


class RouterState(GraphState):
    result: LastValue[dict]  # Store the result from routes
    execution_order: History[str]  # Track execution order

@pytest.mark.asyncio
async def test_async_parallel_execution():
    state = StateForTestParallelExecution(execution_order=[])
    basic_graph = Graph(state=state)

    @basic_graph.node()
    async def task1(state):
        return {"execution_order": "task1"}

    @basic_graph.node()
    async def task2(state):
        return {"execution_order": "task2"}

    @basic_graph.node()
    async def task3(state):
        return {"execution_order": "task3"}

    basic_graph.add_edge(START, "task1")
    basic_graph.add_edge("task1", "task2")
    basic_graph.add_edge("task1", "task3")
    basic_graph.add_edge("task2", END)
    basic_graph.add_edge("task3", END)
    basic_graph.compile()

    # Execute the graph
    await basic_graph.start_async()

    # Verify task1 was executed first
    assert state.execution_order[0] == "task1"

    # Verify task2 and task3 were both executed after task1
    assert set(state.execution_order[1:]) == {"task2", "task3"}
    assert len(state.execution_order) == 3


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
        await basic_graph.start_async()

    assert "Task failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_parallel_execution_timeout():
    basic_graph = Graph()

    @basic_graph.node()
    async def slow_task():
        await asyncio.sleep(3)  # Task that takes too long

    @basic_graph.node()
    async def normal_task():
        pass

    basic_graph.add_edge(START, "slow_task")
    basic_graph.add_edge("slow_task", "normal_task")
    basic_graph.add_edge("normal_task", END)
    basic_graph.compile()

    # Verify timeout error is raised
    with pytest.raises(TimeoutError) as exc_info:
        await basic_graph.start_async(timeout=1)

    assert "Execution timeout" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_pause_before_node_execution():
    state = StateForTestParallelExecution(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="before")
    async def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    async def task3(state):
        return {"execution_order": "task3"}

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", END)
    graph.compile()

    # First execution should stop before task2
    await graph.start_async()
    assert state.execution_order == ["task1"]
    assert graph.next_execution_node == "task2"

    # Resume execution
    await graph.resume_async()
    assert state.execution_order == ["task1", "task2", "task3"]


@pytest.mark.asyncio
async def test_async_pause_after_node_execution():
    state = StateForTestParallelExecution(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="after")
    async def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    async def task3(state):
        return {"execution_order": "task3"}

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", END)
    graph.compile()

    # First execution should stop after task2
    await graph.start_async()
    assert state.execution_order == ["task1", "task2"]
    assert graph.next_execution_node == "task3"

    # Resume execution
    await graph.resume_async()
    assert state.execution_order == ["task1", "task2", "task3"]


@pytest.mark.asyncio
async def test_async_resume_without_pause():
    graph = Graph()

    @graph.node()
    async def task1():
        pass

    @graph.node()
    async def task2():
        pass

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", END)
    graph.compile()

    # Should raise error when trying to resume without a pause
    with pytest.raises(ValueError):
        await graph.resume_async()


@pytest.mark.asyncio
async def test_async_multiple_pause_resume_cycles():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="after")
    async def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    async def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    async def task4(state):
        return {"execution_order": "task4"}

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # First execution - stops after task2
    await graph.start_async()
    assert graph.state.execution_order == ["task1", "task2"]
    assert graph.next_execution_node == "task3"

    # Second resume - completes execution
    await graph.resume_async()
    assert graph.state.execution_order == ["task1", "task2", "task3", "task4"]


@pytest.mark.asyncio
async def test_async_pause_resume_with_parallel_execution():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def task1(state):
        return {"execution_order": "task1"}

    @graph.node()
    async def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    async def task3(state):
        return {"execution_order": "task3"}

    @graph.node(interrupt="before")
    async def task4(state):
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
    await graph.start_async()
    assert "task1" in graph.state.execution_order
    assert "task3" in graph.state.execution_order
    assert "task2" in graph.state.execution_order
    assert "task4" not in graph.state.execution_order
    assert graph.next_execution_node == "task4"

    # Resume should complete the execution
    await graph.resume_async()
    assert "task1" in graph.state.execution_order
    assert "task3" in graph.state.execution_order
    assert "task2" in graph.state.execution_order
    assert "task4" in graph.state.execution_order


@pytest.mark.asyncio
async def test_async_parallel_updates():
    state = StateForTestWithBuffers(counter=0, status="", metrics=[])
    graph = Graph(state=state)

    @graph.node()
    async def increment_counter(state):
        return {"counter": 1}

    @graph.node()
    async def update_status(state):
        return {"status": "running"}

    @graph.node()
    async def add_metrics_1(state):
        return {"metrics": {"accuracy": 0.95}}

    @graph.node()
    async def add_metrics_2(state):
        return {"metrics": {"precision": 0.90}}

    @graph.node()
    async def add_metrics_3(state):
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

    # Execute the graph multiple times
    for _ in range(3):
        await graph.start_async()

    # Check the state after execution
    state = graph.state

    # Verify that the counter was incremented correctly
    assert state.counter == 1

    # Verify that the status was updated
    assert state.status == "running"

    # Verify that metrics were added correctly
    assert len(state.metrics) == 3
    expected_metrics = [
        {"accuracy": 0.95},
        {"precision": 0.90},
        {"recall": 0.85},
    ]
    for metric in state.metrics:
        assert metric in expected_metrics


@pytest.mark.asyncio
async def test_async_initial_state_with_filled_values():
    state = StateForTestWithHistoryIncremental(
        execution_order=["pre_task", "task0"], counter=2
    )
    graph = Graph(state=state)

    @graph.node()
    async def task1(state):
        return {"execution_order": "task1", "counter": 3}

    @graph.node()
    async def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    async def task3(state):
        return {"execution_order": "task3", "counter": 4}

    @graph.node()
    async def task4(state):
        return {"execution_order": "task4"}

    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # Start execution from task2
    await graph.start_async()
    expected_tasks = {"pre_task", "task0", "task1", "task2", "task3", "task4"}
    assert set(graph.state.execution_order) == expected_tasks
    assert graph.state.counter == 9  # 2 + 3 + 4


@pytest.mark.asyncio
async def test_async_state_modification_during_execution():
    state = StateForTestWithHistory(execution_order=[])
    graph = Graph(state=state)

    @graph.node()
    async def task1(state):
        return {"execution_order": "task1"}

    @graph.node(interrupt="after")
    async def task2(state):
        return {"execution_order": "task2"}

    @graph.node()
    async def task3(state):
        return {"execution_order": "task3"}

    @graph.node()
    async def task4(state):
        return {"execution_order": "task4"}

    # Create parallel paths
    graph.add_edge(START, "task1")
    graph.add_edge("task1", "task2")
    graph.add_edge("task2", "task3")
    graph.add_edge("task3", "task4")
    graph.add_edge("task4", END)
    graph.compile()

    # First execution should execute task1 and task2, then pause
    await graph.start_async()
    assert "task1" in graph.state.execution_order
    assert "task2" in graph.state.execution_order
    assert "task3" not in graph.state.execution_order
    assert "task4" not in graph.state.execution_order

    state.execution_order.append("appended_value")
    assert state.execution_order == ["task1", "task2", "appended_value"]

    # Resume should complete the execution
    await graph.resume_async()
    assert graph.state.execution_order == [
        "task1",
        "task2",
        "appended_value",
        "task3",
        "task4",
    ]


@pytest.mark.asyncio
async def test_async_execution_steps_with_interrupt():
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
    async def task1(state):
        return {
            "number_of_executed_steps": 1,
            "current_status": "task1_complete"
        }

    @graph.node(interrupt="before")
    async def task2(state):
        return {
            "number_of_executed_steps": 1,
            "current_status": "task2_complete"
        }

    @graph.node()
    async def task3(state):
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
    await graph.start_async()
    assert graph.state.number_of_executed_steps == 1  # Only task1 executed
    assert graph.state.current_status == "task1_complete"
    assert graph.next_execution_node == "task2"

    # Resume execution - should complete remaining tasks
    await graph.resume_async()
    assert graph.state.number_of_executed_steps == 3  # All tasks executed
    assert graph.state.current_status == "task3_complete"


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
    await graph.start_async()
    assert state.result == {"result": "from route A"}
    assert state.execution_order == ["route_a"]

    # First resume - should execute route_b and pause before route_b again
    await graph.resume_async()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b"]

    # Second resume - should execute route_b and pause before route_b again
    await graph.resume_async()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b"]

    # Third resume - pattern continues
    await graph.resume_async()
    assert state.result == {"result": "from route B"}
    assert state.execution_order == ["route_a", "route_b", "route_b", "route_b"]
