import time
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


def test_sequential_repeated_nodes():
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

  graph.start()

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


def test_parallel_repeated_nodes():
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
  graph.start()
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


def test_mixed_repeated_nodes():
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

  graph.start()

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


def test_error_handling_in_repeated_nodes():
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
    graph.start()
