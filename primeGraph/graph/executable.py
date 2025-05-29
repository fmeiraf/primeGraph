import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union

from pydantic import BaseModel

from primeGraph.buffer.base import BaseBuffer
from primeGraph.buffer.factory import BufferFactory
from primeGraph.checkpoint.base import CheckpointData, StorageBackend
from primeGraph.graph.base import BaseGraph
from primeGraph.graph.engine import Engine
from primeGraph.models.checkpoint import Checkpoint
from primeGraph.models.state import GraphState
from primeGraph.types import ChainStatus

if TYPE_CHECKING:
    from primeGraph.graph.engine import Engine


class ExecutableNode(NamedTuple):
    node_name: str
    task_list: List[Union[Callable, "ExecutableNode"]]
    node_list: List[str]
    execution_type: Literal["sequential", "parallel"]
    interrupt: Union[Literal["before", "after"], None] = None


class Graph(BaseGraph):
    def __init__(
        self,
        state: Union[GraphState, None] = None,
        checkpoint_storage: Optional[StorageBackend] = None,
        chain_id: Optional[str] = None,
        execution_timeout: Union[int] = 60 * 5,
        max_node_iterations: int = 100,
        verbose: bool = False,
    ):
        super().__init__(state)

        # Configure logging based on verbose flag
        self.verbose = verbose
        if not verbose:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)
            logging.getLogger("graphviz").setLevel(logging.WARNING)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            logging.getLogger("graphviz").setLevel(logging.WARNING)

        # State management
        self.initial_state = state
        self.state: Union[GraphState, None] = state
        self.state_schema = self._get_schema(state)
        self.buffers: Dict[str, BaseBuffer] = {}
        if self.state_schema:
            self._assign_buffers()
            self._update_buffers_from_state()

        # Chain management
        self.chain_id = chain_id if chain_id else f"chain_{uuid.uuid4()}"
        self.checkpoint_storage = checkpoint_storage
        self.chain_status = ChainStatus.IDLE

        # Execution management
        self.execution_timeout = execution_timeout
        self.max_node_iterations = max_node_iterations
        self.execution_engine: Engine = Engine(
            graph=self, timeout=execution_timeout, max_node_iterations=max_node_iterations
        )

    def _assign_buffers(self) -> None:
        if not self.state_schema:
            raise ValueError("No state schema found. Please set state.")

        self.buffers = {
            field_name: BufferFactory.create_buffer(field_name, field_type)
            for field_name, field_type in self.state_schema.items()
        }

    def _reset_state(self, new_state: Union[BaseModel, None] = None) -> Union[BaseModel, None]:
        """Reset the state instance to its initial values while preserving the class."""
        if not self.initial_state:
            return None

        if new_state:
            # Update both state and initial_state with the new values
            new_state_dict = new_state.model_dump()
            self.initial_state = self.initial_state.__class__(**new_state_dict)
            return self.initial_state.__class__(**new_state_dict)
        else:
            # Reset to initial values
            initial_state_dict = self.initial_state.model_dump()
            return self.initial_state.__class__(**initial_state_dict)

    def _get_schema(self, state: Union[BaseModel, NamedTuple, None]) -> Union[Dict[str, Any], Any]:
        if isinstance(state, (BaseModel, GraphState)) and hasattr(state, "get_buffer_types"):
            return state.get_buffer_types()
        elif isinstance(state, tuple) and hasattr(state, "_fields"):
            return state.__annotations__
        return None

    def _get_chain_status(self) -> ChainStatus:
        return self.chain_status

    def _clean_graph_variables(self, new_state: Union[BaseModel, None] = None) -> None:
        self.chain_status = ChainStatus.IDLE

        # Re-assign buffers and reset state
        if self.state_schema:
            self._assign_buffers()
            self._update_buffers_from_state()  # Update buffers with current state values

        # Don't reset state to initial values
        # self._reset_state(new_state)  # Comment out or remove this line

    def _update_chain_status(self, status: ChainStatus) -> None:
        self.chain_status = status
        if self.verbose:
            self.logger.debug(f"Chain status updated to: {status}")

    def _update_state_from_buffers(self) -> None:
        for field_name, buffer in self.buffers.items():
            if buffer._ready_for_consumption:
                setattr(self.state, field_name, buffer.consume_last_value())

    def _update_buffers_from_state(self) -> None:
        for field_name, buffer in self.buffers.items():
            buffer.set_value(getattr(self.state, field_name))

    def _save_checkpoint(self, node_name: str, engine_state: Dict[str, Any]) -> None:
        if self.state and self.checkpoint_storage:
            checkpoint_data = CheckpointData(
                chain_id=self.chain_id,
                chain_status=self.chain_status,
                engine_state=engine_state,
            )
            self.checkpoint_storage.save_checkpoint(
                state_instance=self.state,
                checkpoint_data=checkpoint_data,
            )
        if self.verbose:
            self.logger.debug(f"Checkpoint saved after node: {node_name}")

    async def execute(
        self,
        timeout: Union[int, None] = None,
        chain_id: Optional[str] = None,
    ) -> str:
        """
        Start a new execution of the graph.

        Args:
            start_from: Optional node name to start execution from.
            timeout: Optional timeout for the execution.
            chain_id: Optional chain ID for the execution.
        """
        if chain_id:
            self.chain_id = chain_id

        if not timeout:
            timeout = self.execution_timeout

        self._clean_graph_variables()

        # always start a new engine when starting a new execution
        self.execution_engine = Engine(graph=self, timeout=timeout, max_node_iterations=self.max_node_iterations)
        await self.execution_engine.execute()
        return self.chain_id

    async def resume(
        self, timeout: Optional[Union[int, float]] = None, max_node_iterations: Optional[int] = None
    ) -> str:
        await self.execution_engine.resume()
        return self.chain_id

    def load_from_checkpoint(self, chain_id: str, checkpoint_id: Optional[str] = None) -> None:
        """Load graph state and execution variables from a checkpoint.

        Args:
            checkpoint_id: Optional specific checkpoint ID to load. If None, loads the last checkpoint.
        """
        if not self.checkpoint_storage:
            raise ValueError("Checkpoint storage must be configured to load from checkpoint")

        # Get checkpoint ID if not specified
        if not checkpoint_id:
            checkpoint_id = self.checkpoint_storage.get_last_checkpoint_id(chain_id)
            if not checkpoint_id:
                raise ValueError(f"No checkpoints found for chain {chain_id}")

        # Load checkpoint data
        if self.state:
            checkpoint: Checkpoint = self.checkpoint_storage.load_checkpoint(
                state_instance=self.state,
                chain_id=chain_id,
                checkpoint_id=checkpoint_id,
            )

        # Verify state class matches
        current_state_class = f"{self.state.__class__.__module__}.{self.state.__class__.__name__}"
        if current_state_class != checkpoint.state_class:
            raise ValueError(
                f"State class mismatch. Current: {current_state_class}, " f"Checkpoint: {checkpoint.state_class}"
            )

        self._clean_graph_variables()
        # Update state from serialized data
        if self.initial_state:
            self.state = self.initial_state.__class__.model_validate_json(checkpoint.data)

        # Update buffers with current state values
        for field_name, buffer in self.buffers.items():
            buffer.set_value(getattr(self.state, field_name))

        # Update execution variables
        if checkpoint:
            self.chain_id = checkpoint.chain_id
            self.chain_status = checkpoint.chain_status

            if checkpoint.engine_state:
                self.execution_engine.load_full_state(checkpoint.engine_state)
            else:
                self.logger.warning("No engine state found in checkpoint")

        self.logger.debug(f"Loaded checkpoint {checkpoint_id} for chain {self.chain_id}")

    def set_state_and_checkpoint(self, new_state: Union[BaseModel, Dict[str, Any]]) -> None:
        """Set the graph state and save a checkpoint - ignores buffer type logic.

        Args:
            new_state: New state instance or dict with updates. If dict, only specified fields will be updated.
                      Must match the graph's state schema.
        """
        if not self.state_schema:
            raise ValueError("Graph was initialized without state schema, cannot update state")

        if not self.state:
            raise ValueError("Graph state is not initialized")

        # Handle partial updates via dict
        if isinstance(new_state, dict):
            # Validate that all keys exist in current state
            invalid_keys = set(new_state.keys()) - set(self.state.model_fields.keys())
            if invalid_keys:
                raise ValueError(f"Invalid state fields: {invalid_keys}")

            # Create updated state by merging current state with updates
            current_state_dict = self.state.model_dump()
            current_state_dict.update(new_state)
            new_state = self.state.__class__(**current_state_dict)

        # Handle full state update via BaseModel
        elif not isinstance(new_state, self.state.__class__):
            raise ValueError(f"New state must be an instance of {self.state.__class__.__name__} or a dict")

        # Update state
        self.state = new_state
        self._update_buffers_from_state()

        # Save checkpoint
        if self.checkpoint_storage:
            checkpoint_data = CheckpointData(
                chain_id=self.chain_id,
                chain_status=self.chain_status,
            )
            self.checkpoint_storage.save_checkpoint(
                state_instance=self.state,
                checkpoint_data=checkpoint_data,
            )
            if self.verbose:
                self.logger.debug("Checkpoint saved after state update")

        elif self.verbose:
            self.logger.debug("No checkpoint storage configured - state updated without checkpoint")

    def update_state_and_checkpoint(self, updates: Dict[str, Any]) -> None:
        """Update specific state fields according to their buffer types and save a checkpoint.

        Args:
            updates: Dict of field updates. Values will be handled according to their buffer type:
                    - History: Values will be appended to existing list
                    - Incremental: Values will be added to current value
                    - LastValue: Values will replace current value
        """
        if not self.state_schema:
            raise ValueError("Graph was initialized without state schema, cannot update state")

        if not self.state:
            raise ValueError("Graph state is not initialized")

        # Validate that all keys exist in current state
        invalid_keys = set(updates.keys()) - set(self.state.model_fields.keys())
        if invalid_keys:
            raise ValueError(f"Invalid state fields: {invalid_keys}")

        # Update each field according to its buffer type
        execution_id = f"update_{uuid.uuid4().hex[:8]}"
        for field_name, new_value in updates.items():
            buffer = self.buffers[field_name]

            # Update buffer with new value
            buffer.update(new_value, execution_id)

        # Update state from buffers
        self._update_state_from_buffers()

        # Save checkpoint
        if self.checkpoint_storage:
            checkpoint_data = CheckpointData(
                chain_id=self.chain_id,
                chain_status=self.chain_status,
                engine_state=self.execution_engine.get_full_state(),
            )
            self.checkpoint_storage.save_checkpoint(
                state_instance=self.state,
                checkpoint_data=checkpoint_data,
            )
            if self.verbose:
                self.logger.debug("Checkpoint saved after state update")

        elif self.verbose:
            self.logger.debug("No checkpoint storage configured - state updated without checkpoint")
