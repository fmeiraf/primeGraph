from typing import Optional

import pytest

from primeGraph.buffer.factory import History, LastValue
from primeGraph.checkpoint.base import CheckpointData
from primeGraph.checkpoint.local_storage import LocalStorage
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState


class StateForTest(GraphState):
  value: LastValue[int]
  text: LastValue[Optional[str]] = None


def test_save_and_load_checkpoint():
  # Initialize
  state = StateForTest(value=42, text="initial")
  graph = Graph(state=state, checkpoint_storage=LocalStorage())

  # Save checkpoint using CheckpointData
  checkpoint_data = CheckpointData(
    chain_id=graph.chain_id,
    chain_status=graph.chain_status,
  )
  checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  # Load checkpoint
  loaded_state = graph.checkpoint_storage.load_checkpoint(state, graph.chain_id, checkpoint_id)

  serialized_data = state.__class__.model_validate_json(loaded_state.data)

  assert serialized_data.value == state.value
  assert serialized_data.text == state.text


def test_list_checkpoints():
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=LocalStorage())

  # Save multiple checkpoints using CheckpointData
  checkpoint_data = CheckpointData(
    chain_id=graph.chain_id,
    chain_status=graph.chain_status,
  )
  checkpoint_1 = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  state.value = 43
  checkpoint_2 = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  checkpoints = graph.checkpoint_storage.list_checkpoints(graph.chain_id)
  assert len(checkpoints) == 2
  assert checkpoint_1 in [c.checkpoint_id for c in checkpoints]
  assert checkpoint_2 in [c.checkpoint_id for c in checkpoints]


def test_delete_checkpoint():
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=LocalStorage())

  checkpoint_data = CheckpointData(
    chain_id=graph.chain_id,
    chain_status=graph.chain_status,
  )
  checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)
  assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 1

  graph.checkpoint_storage.delete_checkpoint(graph.chain_id, checkpoint_id)
  assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 0


def test_version_mismatch():
  class NewStateForTest(GraphState):
    value: LastValue[int]
    text: LastValue[Optional[str]] = None
    new_value: LastValue[int]  # new attribute

  # Save with original version
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=LocalStorage())

  checkpoint_data = CheckpointData(
    chain_id=graph.chain_id,
    chain_status=graph.chain_status,
  )
  checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  # Try to save with new version
  with pytest.raises(ValueError):
    graph.checkpoint_storage.load_checkpoint(NewStateForTest, graph.chain_id, checkpoint_id)


def test_nonexistent_checkpoint():
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=LocalStorage())

  with pytest.raises(KeyError):
    graph.checkpoint_storage.load_checkpoint(state, graph.chain_id, "nonexistent")


def test_nonexistent_chain():
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=LocalStorage())

  with pytest.raises(KeyError):
    graph.checkpoint_storage.load_checkpoint(state, "nonexistent", "some_checkpoint")


class StateForTestWithHistory(GraphState):
  execution_order: History[str]


def test_resume_with_checkpoint_load():
  state = StateForTestWithHistory(execution_order=[])
  storage = LocalStorage()
  graph = Graph(state=state, checkpoint_storage=storage)

  @graph.node()
  def task1(state):
    print("task1")
    return {"execution_order": "task1"}

  @graph.node()
  def task2(state):
    print("task2")
    return {"execution_order": "task2"}

  @graph.node()
  def task3(state):
    print("task3")
    return {"execution_order": "task3"}

  @graph.node()
  def task4(state):
    print("task4")

    return {"execution_order": "task4"}

  @graph.node()
  def task5(state):
    print("task5")
    return {"execution_order": "task5"}

  @graph.node(interrupt="before")
  def task6(state):
    print("task6")
    return {"execution_order": "task6"}

  graph.add_edge(START, "task1")
  graph.add_edge("task1", "task2")
  graph.add_edge("task2", "task3")
  graph.add_edge("task2", "task4")
  graph.add_edge("task2", "task5")
  graph.add_edge("task4", "task6")
  graph.add_edge("task3", "task6")
  graph.add_edge("task5", "task6")
  graph.add_edge("task6", END)
  graph.compile()

  chain_id = graph.start()
  assert all(
    task in graph.state.execution_order for task in ["task1", "task2", "task3", "task4", "task5"]
  ), "tasks are not in there"
  assert len(storage.list_checkpoints(graph.chain_id)) == 4  # n + 1 due to interrupt

  # start a new chain just to test the load from checkpoint
  new_chain_id = graph.start()
  assert new_chain_id != chain_id

  # loading first chain state
  graph.load_from_checkpoint(chain_id)

  # resuming execution
  graph.resume()
  assert all(
    task in graph.state.execution_order for task in ["task1", "task2", "task3", "task4", "task5", "task6"]
  ), "tasks are not in there"
