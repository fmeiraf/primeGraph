import time
from time import sleep
from typing import List

import pytest

from primeGraph.buffer.factory import History, Incremental
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState


class StateWithHistory(GraphState):
  execution_order: History[str]
  execution_times: History[float]
  counter: Incremental[int]


@pytest.mark.asyncio
async def test_sequential_repeated_nodes():
  state = StateWithHistory(execution_order=[], execution_times=[], counter=0)
  graph = Graph(state=state)

  @graph.node()
  def start_task(state):
    return {}

  @graph.node()
  def repeated_task(state):
    time.sleep(0.1)  # Simulate some work
    current_time = time.time()
    return {
      "execution_order": f"task_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  # Create a sequential chain with 3 repetitions
  graph.add_edge(START, "start_task")
  graph.add_repeating_edge("start_task", "repeated_task", END, repeat=3, parallel=False)
  graph.compile()

  await graph.execute()

  # Verify execution order
  assert len(state.execution_order) == 3
  assert state.execution_order == ["task_0", "task_1", "task_2"]

  # Verify sequential execution by checking timestamps
  execution_times: List[float] = state.execution_times
  for i in range(1, len(execution_times)):
    time_diff = execution_times[i] - execution_times[i - 1]
    assert time_diff >= 0.1  # Each task should take at least 0.1s

  # Verify total counter value
  assert state.counter == 3  # Each task added 1


@pytest.mark.asyncio
async def test_parallel_repeated_nodes():
  state = StateWithHistory(execution_order=[], execution_times=[], counter=0)
  graph = Graph(state=state)

  @graph.node()
  def start_task(state):
    return {}

  @graph.node()
  def repeated_task(state):
    time.sleep(0.1)  # Simulate some work
    current_time = time.time()
    return {
      "execution_order": f"task_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  # Create a parallel execution with 3 repetitions
  graph.add_edge(START, "start_task")
  graph.add_repeating_edge("start_task", "repeated_task", END, repeat=3, parallel=True)
  graph.compile()

  start_time = time.time()
  await graph.execute()
  total_time = time.time() - start_time

  # Verify all tasks were executed
  assert len(state.execution_order) == 3

  # Verify parallel execution by checking timestamps
  execution_times: List[float] = state.execution_times
  max_time_diff = max(abs(t2 - t1) for t1, t2 in zip(execution_times[:-1], execution_times[1:], strict=False))
  assert max_time_diff < 0.1  # Tasks should complete very close to each other

  # Total execution time should be close to a single task execution
  assert 0.1 <= total_time < 0.2

  # Verify total counter value
  assert state.counter == 3  # Each task added 1


@pytest.mark.asyncio
async def test_mixed_repeated_nodes():
  state = StateWithHistory(execution_order=[], execution_times=[], counter=0)
  graph = Graph(state=state)

  @graph.node()
  def start_task(state):
    return {}

  @graph.node()
  def sequential_task(state):
    time.sleep(0.1)
    current_time = time.time()
    return {
      "execution_order": f"seq_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  @graph.node()
  def intermediate_task(state):
    time.sleep(0.1)
    return {}

  @graph.node()
  def parallel_task(state):
    time.sleep(0.1)
    current_time = time.time()
    return {
      "execution_order": f"par_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  # Create a mixed execution pattern
  graph.add_edge(START, "start_task")
  graph.add_repeating_edge("start_task", "sequential_task", "intermediate_task", repeat=2, parallel=False)
  graph.add_repeating_edge("intermediate_task", "parallel_task", END, repeat=3, parallel=True)
  graph.compile()

  await graph.execute()

  # Verify execution count
  assert len(state.execution_order) == 5  # 2 sequential + 3 parallel

  # Verify sequential part executed sequentially
  sequential_times = state.execution_times[:3]
  for i in range(1, len(sequential_times)):
    time_diff = sequential_times[i] - sequential_times[i - 1]
    assert time_diff >= 0.1

  # Verify parallel part executed in parallel
  parallel_times = state.execution_times[3:]
  max_parallel_diff = max(abs(t2 - t1) for t1, t2 in zip(parallel_times[:-1], parallel_times[1:], strict=False))
  assert max_parallel_diff < 0.1

  # Verify counter
  assert state.counter == 5  # Total of 6 tasks executed


@pytest.mark.asyncio
async def test_error_handling_in_repeated_nodes():
  graph = Graph()

  @graph.node()
  def start_task():
    return {}

  @graph.node()
  def failing_task():
    raise ValueError("Task failed")

  # Try to create invalid repetition
  with pytest.raises(ValueError):
    graph.add_repeating_edge("failing_task", "failing_task", END, repeat=0, parallel=True)

  # Set up valid edges
  graph.add_edge(START, "start_task")
  graph.add_repeating_edge("start_task", "failing_task", END, repeat=3, parallel=True)
  graph.compile()

  # Verify error propagation
  with pytest.raises(RuntimeError):
    await graph.execute()


@pytest.mark.asyncio
async def test_repeated_nodes_execution_count():
    """
    Test that a repeating edge executes the source node (Node A) and sink (Node C)
    once, and the repeated node (Node B) the expected number of times (3),
    while also verifying that custom node names were assigned.
    """
    # Define a custom state with an execution log.
    class ExecutionState(GraphState):
        execution_log: History[str]

    state = ExecutionState(execution_log=[])
    graph = Graph(state=state)

    @graph.node()
    def node_a(state: ExecutionState):
        return {"execution_log": "node_a"}

    @graph.node()
    def node_b(state: ExecutionState):
        return {"execution_log": "node_b"}

    @graph.node()
    def node_c(state: ExecutionState):
        return {"execution_log": "node_c"}

    # Create a repeating edge from node_a through node_b (3 times) to node_c,
    # with custom names for the repeated nodes.
    graph.add_edge(START, "node_a")
    graph.add_repeating_edge(
        "node_a",
        "node_b",
        "node_c",
        repeat=3,
        parallel=False,
        repeat_names=["node_b_1", "node_b_2", "node_b_3"]
    )
    graph.add_edge("node_c", END)
    graph.compile()

    # Verify that the custom repeat node names were created in the graph
    node_ids = graph.nodes.keys()
    assert "node_b_1" in node_ids, "Expected custom node 'node_b_1' not found."
    assert "node_b_2" in node_ids, "Expected custom node 'node_b_2' not found."
    assert "node_b_3" in node_ids, "Expected custom node 'node_b_3' not found."

    # Execute the graph.
    await graph.execute()

    # Check that Node A executes once, Node B executes three times, and Node C executes once.
    expected_log = ["node_a", "node_b", "node_b", "node_b", "node_c"]
    assert state.execution_log == expected_log, f"Unexpected execution order: {state.execution_log}"


@pytest.mark.asyncio
async def test_repeated_nodes_execution_count_parallel():
    """
    Test that a repeating edge executes the source node (Node A) and sink (Node C)
    once, and the repeated node (Node B) the expected number of times (3) in parallel,
    while also verifying that custom node names were assigned.
    """
    # Define a custom state with an execution log.
    class ExecutionState(GraphState):
        execution_log: History[str]

    state = ExecutionState(execution_log=[])
    graph = Graph(state=state)

    @graph.node()
    def node_a(state: ExecutionState):
        return {"execution_log": "node_a"}

    @graph.node()
    def node_b(state: ExecutionState):
        return {"execution_log": "node_b"}

    @graph.node()
    def node_c(state: ExecutionState):
        return {"execution_log": "node_c"}

    # Create a repeating edge from node_a through node_b (3 times) to node_c,
    # with custom names for the repeated nodes and parallel execution.
    graph.add_edge(START, "node_a")
    graph.add_repeating_edge(
        "node_a",
        "node_b",
        "node_c",
        repeat=3,
        parallel=True,
        repeat_names=["node_b_1", "node_b_2", "node_b_3"]
    )
    graph.add_edge("node_c", END)
    graph.compile()

    # Verify that the custom repeat node names were created in the graph.
    node_ids = graph.nodes.keys()
    assert "node_b_1" in node_ids, "Expected custom node 'node_b_1' not found."
    assert "node_b_2" in node_ids, "Expected custom node 'node_b_2' not found."
    assert "node_b_3" in node_ids, "Expected custom node 'node_b_3' not found."

    # Execute the graph.
    await graph.execute()

    # Check that Node A executes once, Node B executes three times, and Node C executes once.
    # In parallel execution, although the order of Node B executions may not be strictly sequential,
    # Node A should be the first and Node C should be the last in the execution log.
    expected_length = 5  # 1 (node_a) + 3 (node_b) + 1 (node_c)
    assert len(state.execution_log) == expected_length, f"Expected execution log length {expected_length}, got {len(state.execution_log)}"
    assert state.execution_log[0] == "node_a", f"Expected the first log entry to be 'node_a', got {state.execution_log[0]}"
    assert state.execution_log[-1] == "node_c", f"Expected the last log entry to be 'node_c', got {state.execution_log[-1]}"

    # Check the three middle log entries are all 'node_b'
    for log_entry in state.execution_log[1:-1]:
        assert log_entry == "node_b", f"Expected 'node_b', got {log_entry}"


@pytest.mark.asyncio
async def test_repeated_nodes_execution_count_parallel_with_timestamps():
    """
    Test that a repeating edge executes the source node (Node A) and sink (Node C)
    once, and the repeated node (Node B) three times in parallel by checking their timestamps.
    """
    # Define a custom state with an execution log and timestamps.
    class ExecutionState(GraphState):
        execution_log: History[str]
        timestamps: History[float]

    state = ExecutionState(execution_log=[], timestamps=[])
    graph = Graph(state=state)

    @graph.node()
    def node_a(state: ExecutionState):
        return {"execution_log": "node_a"}

    @graph.node()
    def node_b(state: ExecutionState):
        # Simulate work with a sleep.
        sleep(0.1)
        current_time = time.time()
        return {"execution_log": "node_b", "timestamps": current_time}

    @graph.node()
    def node_c(state: ExecutionState):
        return {"execution_log": "node_c"}

    # Create a repeating edge from node_a through node_b (3 times) to node_c with parallel execution.
    graph.add_edge(START, "node_a")
    graph.add_repeating_edge(
        "node_a",
        "node_b",
        "node_c",
        repeat=3,
        parallel=True,
        repeat_names=["node_b_1", "node_b_2", "node_b_3"]
    )
    graph.add_edge("node_c", END)
    graph.compile()

    start_time = time.time()
    await graph.execute()
    total_time = time.time() - start_time

    # Verify the execution log contains node_a, three instances of node_b and node_c in order.
    expected_length = 5  # 1 (node_a) + 3 (node_b) + 1 (node_c)
    assert len(state.execution_log) == expected_length, \
        f"Expected log length {expected_length}, got {len(state.execution_log)}"
    assert state.execution_log[0] == "node_a", "First log entry should be 'node_a'"
    assert state.execution_log[-1] == "node_c", "Last log entry should be 'node_c'"
    for log_entry in state.execution_log[1:-1]:
        assert log_entry == "node_b", f"Expected 'node_b', got {log_entry}"

    # Check that the node_b executions happened concurrently.
    # In parallel execution, the node_b tasks should have overlapping timestamps.
    # The maximum difference between the earliest and latest timestamp should be significantly less than each node's sleep duration (0.1s).
    max_timestamp_diff = max(state.timestamps) - min(state.timestamps)
    assert max_timestamp_diff < 0.05, f"Tasks did not run concurrently, timestamp diff was {max_timestamp_diff:.3f}s"

    # Additionally, the overall execution time should be close to the duration of a single node_b task,
    # rather than triple the duration if the tasks were sequential.
    assert total_time < 0.2, f"Total execution time {total_time:.3f}s indicates sequential execution"
