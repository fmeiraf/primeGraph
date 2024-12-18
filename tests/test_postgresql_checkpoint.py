from typing import Optional

import pytest

from primeGraph.buffer.factory import History, LastValue
from primeGraph.checkpoint.base import CheckpointData
from primeGraph.checkpoint.postgresql import PostgreSQLStorage
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState

# Requires you to be running the docker from primeGraph/docker


@pytest.fixture
def postgres_storage():
  storage = PostgreSQLStorage.from_config(
    host="localhost",
    port=5432,
    user="primegraph",
    password="primegraph",
    database="primegraph",
  )
  assert storage.check_schema(), "Schema is not valid"
  return storage


class StateForTest(GraphState):
  value: LastValue[int]
  text: LastValue[Optional[str]] = None


def test_save_and_load_checkpoint(postgres_storage):
  # Initialize
  state = StateForTest(value=42, text="initial")
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  # Save checkpoint
  checkpoint_data = CheckpointData(chain_id=graph.chain_id, chain_status=graph.chain_status)
  checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  # Load checkpoint
  loaded_state = graph.checkpoint_storage.load_checkpoint(state, graph.chain_id, checkpoint_id)

  serialized_data = state.__class__.model_validate_json(loaded_state.data)

  assert serialized_data.value == state.value
  assert serialized_data.text == state.text


def test_list_checkpoints(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  # Save multiple checkpoints
  checkpoint_data = CheckpointData(chain_id=graph.chain_id, chain_status=graph.chain_status)
  checkpoint_1 = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  state.value = 43
  checkpoint_2 = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  checkpoints = graph.checkpoint_storage.list_checkpoints(graph.chain_id)
  assert len(checkpoints) == 2
  assert checkpoint_1 in [c.checkpoint_id for c in checkpoints]
  assert checkpoint_2 in [c.checkpoint_id for c in checkpoints]


def test_delete_checkpoint(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  # Save and then delete a checkpoint
  checkpoint_data = CheckpointData(chain_id=graph.chain_id, chain_status=graph.chain_status)
  checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)
  assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 1

  graph.checkpoint_storage.delete_checkpoint(graph.chain_id, checkpoint_id)
  assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 0


class StateForTestWithHistory(GraphState):
  execution_order: History[str]


def test_resume_with_checkpoint_load(postgres_storage):
  state = StateForTestWithHistory(execution_order=[])
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  @graph.node()
  def task1(state):
    return {"execution_order": "task1"}

  @graph.node()
  def task2(state):
    return {"execution_order": "task2"}

  @graph.node()
  def task3(state):
    return {"execution_order": "task3"}

  @graph.node(interrupt="before")
  def task4(state):
    return {"execution_order": "task4"}

  graph.add_edge(START, "task1")
  graph.add_edge("task1", "task2")
  graph.add_edge("task2", "task3")
  graph.add_edge("task3", "task4")
  graph.add_edge("task4", END)
  graph.compile()

  # Start execution
  chain_id = graph.start()
  assert all(task in graph.state.execution_order for task in ["task1", "task2", "task3"])
  assert len(postgres_storage.list_checkpoints(graph.chain_id)) == 4  # Including interrupt

  # Start new chain to test load
  new_chain_id = graph.start()
  assert new_chain_id != chain_id

  # Load first chain state
  graph.load_from_checkpoint(chain_id)

  # Resume execution
  graph.resume()
  assert all(task in graph.state.execution_order for task in ["task1", "task2", "task3", "task4"])


def test_nonexistent_checkpoint(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  with pytest.raises(KeyError):
    graph.checkpoint_storage.load_checkpoint(state, graph.chain_id, "nonexistent")


def test_nonexistent_chain(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  with pytest.raises(KeyError):
    graph.checkpoint_storage.load_checkpoint(state, "nonexistent", "some_checkpoint")
