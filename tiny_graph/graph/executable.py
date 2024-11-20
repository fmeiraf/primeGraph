import concurrent.futures
import logging
import uuid
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union

from pydantic import BaseModel

from tiny_graph.buffer.base import BaseBuffer
from tiny_graph.buffer.factory import BufferFactory
from tiny_graph.checkpoint.storage.base import StorageBackend
from tiny_graph.graph.base import BaseGraph
from tiny_graph.models.base import GraphState

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ExecutableNode(NamedTuple):
    node_name: str
    task_list: List[Callable]
    execution_type: Literal["sequential", "parallel"]


class ChainStatus(Enum):
    IDLE = auto()
    PAUSE = auto()
    RUNNING = auto()
    FAILED = auto()


class Graph(BaseGraph):
    def __init__(
        self,
        state: Union[BaseModel, NamedTuple, None] = None,
        checkpoint_storage: Optional[StorageBackend] = None,
        chain_id: Optional[str] = None,
    ):
        super().__init__(state)
        self.state = state
        self.state_schema = self._get_schema(state)
        self.state_history = {}
        self.buffers: Dict[str, BaseBuffer] = {}
        if self.state_schema:
            self._assign_buffers()
        self.chain_id = chain_id if chain_id else f"{uuid.uuid4()}"
        self.checkpoint_storage = checkpoint_storage
        self.chain_status = ChainStatus.IDLE

    def _assign_buffers(self):
        self.buffers = {
            field_name: BufferFactory.create_buffer(field_name, field_type)
            for field_name, field_type in self.state_schema.items()
        }

    def _get_schema(self, state: Union[BaseModel, NamedTuple, None]) -> Dict[str, type]:
        if isinstance(state, (BaseModel, GraphState)) and hasattr(
            state, "get_buffer_types"
        ):
            return state.get_buffer_types()
        elif isinstance(state, tuple) and hasattr(state, "_fields"):
            return state.__annotations__
        return None

    def _convert_execution_plan(self) -> List[Any]:
        """Converts the execution plan to a list of functions that have concurrency flags"""

        self._force_compile()

        def create_executable_node(
            exec_plan_item: Union[str, List[Any], None],
        ) -> ExecutableNode:
            if isinstance(exec_plan_item, str):
                return ExecutableNode(
                    node_name=exec_plan_item,
                    task_list=[self.nodes[exec_plan_item].action],
                    execution_type="sequential",
                )
            elif isinstance(exec_plan_item, list):

                def get_group_name(exec_plan_item: List[Any]) -> str:
                    reference_nodes = []
                    for item in exec_plan_item:
                        print(item, type(item))
                        if isinstance(item[0], list):
                            get_group_name(item)
                        elif isinstance(item, str):
                            reference_nodes.append(item)
                        else:
                            reference_nodes.append(item[0])
                    return "_".join(reference_nodes)

                return ExecutableNode(
                    node_name=get_group_name(exec_plan_item),
                    task_list=[create_executable_node(item) for item in exec_plan_item],
                    execution_type="parallel",
                )
            else:
                raise ValueError(f"Expected str or list, got {type(exec_plan_item)}")

        return [create_executable_node(item) for item in self.execution_plan]

    def _update_state_from_buffers(self):
        for field_name, buffer in self.buffers.items():
            if buffer._ready_for_consumption:
                setattr(self.state, field_name, buffer.consume_last_value())
                # Ensure the history key exists and append the new history
                self.state_history.setdefault(f"{field_name}_history", []).append(
                    buffer.value_history
                )

    def _get_chain_status(self) -> ChainStatus:
        return self.chain_status

    def _update_chain_status(self, status: ChainStatus):
        self.chain_status = status

    def execute(
        self, execution_id: str = None, timeout: Union[int, float] = 60 * 5
    ) -> None:
        """Execute the graph with concurrent and sequential execution based on the execution plan.

        Args:
            execution_id: Unique identifier for this execution
        """

        def execute_node(node: ExecutableNode) -> None:
            """Execute a single node or group of nodes with proper concurrency handling."""
            logger.debug(f"Executing node: {node.node_name}")
            if node.execution_type == "sequential":
                for task in node.task_list:
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:

                            def wrapped_task():
                                logger.debug(
                                    f"Executing task in node: {node.node_name}"
                                )
                                result = (
                                    task(state=self.state)
                                    if self._has_state
                                    else task()
                                )
                                # Update buffers with the result if it's returned
                                if result is not None and self._has_state:
                                    for field_name, value in result.items():
                                        if field_name in self.buffers:
                                            self.buffers[field_name].update(
                                                value, execution_id=node.node_name
                                            )
                                return result

                            future = executor.submit(wrapped_task)
                            future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Timeout in sequential node {node.node_name}")
                        raise TimeoutError(
                            f"Execution timeout in sequential node {node.node_name}"
                        )
                    except Exception as e:
                        logger.error(f"Error in node {node.node_name}: {str(e)}")
                        raise RuntimeError(
                            f"Error in node {node.node_name}: {str(e)}"
                        ) from e
            else:  # parallel execution
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for task in node.task_list:
                        if isinstance(task, ExecutableNode):
                            futures.append(executor.submit(execute_node, task))
                        else:

                            def wrapped_task():
                                logger.debug(
                                    f"Executing task in node: {node.node_name}"
                                )
                                result = (
                                    task(state=self.state)
                                    if self._has_state
                                    else task()
                                )
                                # Update buffers with the result if it's returned
                                if result is not None and self._has_state:
                                    for field_name, value in result.items():
                                        if field_name in self.buffers:
                                            self.buffers[field_name].update(
                                                value, execution_id=node.node_name
                                            )
                                return result

                            futures.append(executor.submit(wrapped_task))

                    try:
                        concurrent.futures.wait(
                            futures,
                            timeout=timeout,
                            return_when=concurrent.futures.ALL_COMPLETED,
                        )

                        for future in futures:
                            if future.exception():
                                logger.error(
                                    f"Error in parallel execution of {node.node_name}: {str(future.exception())}"
                                )
                                raise RuntimeError(
                                    f"Error in parallel execution of {node.node_name}: {str(future.exception())}"
                                ) from future.exception()

                    except concurrent.futures.TimeoutError:
                        for future in futures:
                            future.cancel()
                        logger.error(f"Timeout in node {node.node_name}")
                        raise TimeoutError(
                            f"Execution timeout in node {node.node_name}"
                        )

        execution_plan = self._convert_execution_plan()
        for node in execution_plan:
            execute_node(node)
            # Update state after each main-level node execution
            if self.state:
                self._update_state_from_buffers()
                if self.checkpoint_storage:
                    self.checkpoint_storage.save_checkpoint(self.state, self.chain_id)
                logger.debug(f"State updated after node: {node.node_name}")
