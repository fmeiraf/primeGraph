import asyncio
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


@pytest.mark.asyncio
async def test_sequential_repeated_nodes_async():
  state = StateWithHistory(execution_order=[], execution_times=[], counter=0)
  graph = Graph(state=state)

  @graph.node()
  async def start_task(state):
    return {}

  @graph.node()
  async def repeated_task(state):
    await asyncio.sleep(0.1)  # Simulate some work
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

  await graph.start_async()

  # Verify execution order
  assert len(state.execution_order) == 3
  assert state.execution_order == ["task_0", "task_1", "task_2"]

  # Verify sequential execution by checking timestamps
  execution_times: List[float] = state.execution_times
  for i in range(1, len(execution_times)):
    time_diff = execution_times[i] - execution_times[i - 1]
    assert time_diff >= 0.1  # Each task should take at least 0.1s

  # Verify total counter value
  assert state.counter == 3


@pytest.mark.asyncio
async def test_parallel_repeated_nodes_async():
  state = StateWithHistory(execution_order=[], execution_times=[], counter=0)
  graph = Graph(state=state)

  @graph.node()
  async def start_task(state):
    return {}

  @graph.node()
  async def repeated_task(state):
    await asyncio.sleep(0.1)  # Simulate some work
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
  await graph.start_async()
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
  assert state.counter == 3


@pytest.mark.asyncio
async def test_mixed_repeated_nodes_async():
  state = StateWithHistory(execution_order=[], execution_times=[], counter=0)
  graph = Graph(state=state)

  @graph.node()
  async def start_task(state):
    return {}

  @graph.node()
  async def sequential_task(state):
    await asyncio.sleep(0.1)
    current_time = time.time()
    return {
      "execution_order": f"seq_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  @graph.node()
  async def parallel_task(state):
    await asyncio.sleep(0.1)
    current_time = time.time()
    return {
      "execution_order": f"par_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  @graph.node()
  async def intermediate_task(state):
    return {}

  # Create a mixed execution pattern
  graph.add_edge(START, "start_task")
  graph.add_repeating_edge("start_task", "sequential_task", "intermediate_task", repeat=3, parallel=False)
  graph.add_repeating_edge("intermediate_task", "parallel_task", END, repeat=3, parallel=True)
  graph.compile()

  await graph.start_async()

  # Verify execution count
  assert len(state.execution_order) == 6  # 3 sequential + 3 parallel

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
  assert state.counter == 6


@pytest.mark.asyncio
async def test_error_handling_in_repeated_nodes_async():
  graph = Graph()

  @graph.node()
  async def start_task(state):
    return {}

  @graph.node()
  async def failing_task():
    raise ValueError("Task failed")

  # Try to create invalid repetition
  with pytest.raises(ValueError):
    graph.add_repeating_edge("start_task", "failing_task", END, repeat=0, parallel=True)

  # Set up valid edges
  graph.add_edge(START, "start_task")
  graph.add_repeating_edge("start_task", "failing_task", END, repeat=3, parallel=True)
  graph.compile()

  # Verify error propagation
  with pytest.raises(RuntimeError):
    await graph.start_async()


@pytest.mark.asyncio
async def test_large_scale_parallel_performance():
  state = StateWithHistory(execution_order=[], execution_times=[], counter=0)
  graph = Graph(state=state)

  @graph.node()
  async def start_task(state):
    return {}

  @graph.node()
  async def cpu_intensive_task(state):
    # Simulate CPU-intensive work
    await asyncio.sleep(0.1)  # this is needed to simulate i/o blocking tasks
    result = 0
    for _ in range(1000000):
      result += 1
    current_time = time.time()
    return {
      "execution_order": f"task_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  # Create parallel execution with many repetitions
  graph.add_edge(START, "start_task")
  graph.add_repeating_edge("start_task", "cpu_intensive_task", END, repeat=10, parallel=True)
  graph.compile()

  start_time = time.time()
  await graph.start_async()
  parallel_time = time.time() - start_time

  # Now test sequential execution
  graph2 = Graph(state=StateWithHistory(execution_order=[], execution_times=[], counter=0))

  @graph2.node()
  async def start_task_2(state):
    return {}

  @graph2.node()
  async def cpu_intensive_task_2(state):
    # Same task as above
    await asyncio.sleep(0.1)  # this is needed to simulate i/o blocking tasks
    result = 0
    for _ in range(1000000):
      result += 1
    current_time = time.time()
    return {
      "execution_order": f"task_{len(state.execution_order)}",
      "execution_times": current_time,
      "counter": 1,
    }

  graph2.add_edge(START, "start_task_2")
  graph2.add_repeating_edge("start_task_2", "cpu_intensive_task_2", END, repeat=10, parallel=False)
  graph2.compile()

  start_time = time.time()
  await graph2.start_async()
  sequential_time = time.time() - start_time

  # Parallel execution should be significantly faster
  assert parallel_time < sequential_time * 0.7  # At least 30% faster
