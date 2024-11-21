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
from tiny_graph.utils.class_utils import internal_only

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ExecutableNode(NamedTuple):
    node_name: str
    task_list: List[Callable]
    execution_type: Literal["sequential", "parallel"]
    interrupt: Union[Literal["before", "after"], None] = None


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
        execution_timeout: Union[int, float] = 60 * 5,
    ):
        super().__init__(state)

        # State management
        self.state = state
        self.state_schema = self._get_schema(state)
        self.state_history = {}
        self.buffers: Dict[str, BaseBuffer] = {}
        if self.state_schema:
            self._assign_buffers()

        # Chain management
        self.chain_id = chain_id if chain_id else f"{uuid.uuid4()}"
        self.checkpoint_storage = checkpoint_storage
        self.chain_status = ChainStatus.IDLE

        # Execution management
        self.next_execution_node = None
        self.execution_timeout = execution_timeout

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
            # Handle single string nodes (sequential)
            if isinstance(exec_plan_item, str):
                return ExecutableNode(
                    node_name=exec_plan_item,
                    task_list=[self.nodes[exec_plan_item].action],
                    execution_type="sequential",
                    interrupt=self.nodes[exec_plan_item].interrupt,
                )
            # Handle lists (parallel groups)
            elif isinstance(exec_plan_item, list):
                tasks = []
                for item in exec_plan_item:
                    if isinstance(item, str):
                        tasks.append(self.nodes[item].action)
                    elif isinstance(item, list):
                        # Recursively create executable nodes for nested groups
                        tasks.append(create_executable_node(item))
                    else:
                        raise ValueError(f"Expected str or list, got {type(item)}")
                # Join all node names in the task list for the group name
                group_name = "_".join(
                    task.node_name if isinstance(task, ExecutableNode) else item
                    for item, task in zip(exec_plan_item, tasks)
                )
                if any(isinstance(task, ExecutableNode) for task in tasks):
                    return ExecutableNode(
                        node_name=f"sequential_group_{group_name}",
                        task_list=tasks,
                        execution_type="sequential",
                        interrupt=None,
                    )
                else:
                    return ExecutableNode(
                        node_name=f"parallel_group_{group_name}",
                        task_list=tasks,
                        execution_type="parallel",
                        interrupt=None,
                    )
            else:
                raise ValueError(f"Expected str or list, got {type(exec_plan_item)}")

        self.execution_plan = [
            create_executable_node(item) for item in self.execution_path
        ]

    @internal_only
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

    @internal_only
    def _update_chain_status(self, status: ChainStatus):
        self.chain_status = status

    @internal_only
    def _save_state_and_buffers(self, node_name: str):
        if self.state:
            self._update_state_from_buffers()
            if self.checkpoint_storage:
                self.checkpoint_storage.save_checkpoint(self.state, self.chain_id)
        logger.debug(f"State updated after node: {node_name}")

    @internal_only
    def _execute(
        self, start_from: Optional[str] = None, timeout: Union[int, float] = 60 * 5
    ) -> None:
        """Execute the graph with concurrent and sequential execution based on the execution plan.

        Args:
            start_from: node name to start execution from
        """

        def execute_node(
            node: ExecutableNode, start_from: Optional[str] = None
        ) -> None:
            """Execute a single node or group of nodes with proper concurrency handling."""
            if start_from and node.node_name != start_from:
                return

            if node.interrupt == "before":
                if not start_from:
                    self._update_chain_status(ChainStatus.PAUSE)
                    self.next_execution_node = node.node_name
                    return
                else:
                    self.next_execution_node = None

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
                                self.next_execution_node = node.node_name
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
            self.next_execution_node = node.node_name

            if node.interrupt == "after":
                if not start_from:
                    self._update_chain_status(ChainStatus.PAUSE)
                    return

        self._convert_execution_plan()
        self._update_chain_status(ChainStatus.RUNNING)
        for node in self.execution_plan:
            if (
                start_from and node.node_name != start_from
            ):  # when start_from is provided, skip all nodes until the start_from node
                continue

            if (
                start_from
                and node.interrupt == "after"
                and node.node_name == start_from
            ):
                start_from = None
                continue

            chain_status = self._get_chain_status()
            if chain_status == ChainStatus.PAUSE:
                logger.info("Chain paused.")
                return

            execute_node(node, start_from)
            # Update state after each main-level node execution
            self._save_state_and_buffers(node.node_name)

            if start_from and node.node_name == start_from:
                start_from = None

    def execute(
        self, start_from: Optional[str] = None, timeout: Union[int, float, None] = None
    ):
        if timeout is None:
            timeout = self.execution_timeout
        self._execute(start_from, timeout)

    def resume(self):
        if not self.next_execution_node:
            logger.info(
                "No interrupted node found. Starting execution from the beginning."
            )
            raise RuntimeError("No interrupted node found. Cannot resume.")

        self.execute(start_from=self.next_execution_node)
        return
