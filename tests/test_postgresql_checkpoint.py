from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

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


@pytest.mark.asyncio
async def test_resume_with_checkpoint_load(postgres_storage):
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

  
  # Start new chain to test load
  chain_id = await graph.execute()

  # Load first chain state
  graph.load_from_checkpoint(chain_id)

  # Resume execution
  await graph.resume()
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


def test_load_checkpoint_with_nested_basemodels(postgres_storage):
    class BreakdownInstruction(BaseModel):
      step_name: str = Field(description="The name of the step")
      step_instructions: str = Field(description="The instructions for the step")

    class StateWithInstructions(GraphState):
        instructions: LastValue[List[BreakdownInstruction]] = Field(default_factory=list)

    # Create initial state with BreakdownInstruction instances
    instructions = [
        BreakdownInstruction(
            step_name="Research SpaceX Missions",
            step_instructions="Start by researching SpaceX's upcoming missions..."
        ),
        BreakdownInstruction(
            step_name="Budget Planning",
            step_instructions="Calculate the estimated costs..."
        )
    ]
    
    # First graph instance - create and save state
    initial_state = StateWithInstructions(instructions=instructions)
    first_graph = Graph(state=initial_state, checkpoint_storage=postgres_storage)
    
    # Save initial state with checkpoint
    checkpoint_data = CheckpointData(
        chain_id=first_graph.chain_id,
        chain_status=first_graph.chain_status
    )
    first_graph.checkpoint_storage.save_checkpoint(first_graph.state, checkpoint_data)
    chain_id = first_graph.chain_id

    # Create a fresh graph instance with empty state
    fresh_state = StateWithInstructions(instructions=[])
    fresh_graph = Graph(state=fresh_state, checkpoint_storage=postgres_storage)
    
    # Load the checkpoint into the fresh graph
    fresh_graph.load_from_checkpoint(chain_id)

    # Verify the loaded state matches the original
    assert len(fresh_graph.state.instructions) == len(instructions)
    
    for original_instruction, loaded_instruction in zip(instructions, fresh_graph.state.instructions):
        assert isinstance(loaded_instruction, BreakdownInstruction)
        assert loaded_instruction.step_name == original_instruction.step_name
        assert loaded_instruction.step_instructions == original_instruction.step_instructions

    # Verify chain status and ID were properly restored
    assert fresh_graph.chain_id == chain_id
    assert fresh_graph.chain_status == first_graph.chain_status
