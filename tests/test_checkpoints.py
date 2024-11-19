from typing import Optional

import pytest

from tiny_graph.buffer.factory import LastValue
from tiny_graph.checkpoint.storage.local_storage import LocalStorage
from tiny_graph.graph.executable import Graph
from tiny_graph.models.base import GraphState


class StateForTest(GraphState):
    value: LastValue[int]
    text: LastValue[Optional[str]] = None


def test_save_and_load_checkpoint():
    # Initialize
    state = StateForTest(value=42, text="initial")
    graph = Graph(state=state, checkpoint_storage=LocalStorage())

    # Save checkpoint
    checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, graph.chain_id)

    # Load checkpoint
    loaded_state = graph.checkpoint_storage.load_checkpoint(
        state, graph.chain_id, checkpoint_id
    )

    assert loaded_state.value == state.value
    assert loaded_state.text == state.text


def test_list_checkpoints():
    state = StateForTest(value=42)
    graph = Graph(state=state, checkpoint_storage=LocalStorage())

    # Save multiple checkpoints
    checkpoint_1 = graph.checkpoint_storage.save_checkpoint(state, graph.chain_id)
    state.value = 43
    checkpoint_2 = graph.checkpoint_storage.save_checkpoint(state, graph.chain_id)

    checkpoints = graph.checkpoint_storage.list_checkpoints(graph.chain_id)
    assert len(checkpoints) == 2
    assert checkpoint_1 in checkpoints
    assert checkpoint_2 in checkpoints


def test_delete_checkpoint():
    state = StateForTest(value=42)
    graph = Graph(state=state, checkpoint_storage=LocalStorage())

    # Save and then delete a checkpoint
    checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, graph.chain_id)
    assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 1

    graph.checkpoint_storage.delete_checkpoint(checkpoint_id, graph.chain_id)
    assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 0


def test_version_mismatch():
    class NewStateForTest(GraphState):
        value: LastValue[int]
        text: LastValue[Optional[str]] = None
        new_value: LastValue[int]  # new attribute

    # Save with original version
    state = StateForTest(value=42)
    graph = Graph(state=state, checkpoint_storage=LocalStorage())
    checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, graph.chain_id)

    # Try to load with new version
    with pytest.raises(ValueError):
        graph.checkpoint_storage.load_checkpoint(
            NewStateForTest, graph.chain_id, checkpoint_id
        )


def test_nonexistent_checkpoint():
    state = StateForTest(value=42)
    graph = Graph(state=state, checkpoint_storage=LocalStorage())

    with pytest.raises(KeyError):
        graph.checkpoint_storage.load_checkpoint(state, graph.chain_id, "nonexistent")


def test_nonexistent_chain():
    state = StateForTest(value=42)
    graph = Graph(state=state, checkpoint_storage=LocalStorage())

    with pytest.raises(KeyError):
        graph.checkpoint_storage.load_checkpoint(
            state, "nonexistent", "some_checkpoint"
        )
