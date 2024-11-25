import concurrent.futures
import logging
import uuid
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

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
    node_list: List[str]
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
        self.start_from = None
        self.executed_nodes = set()

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
            exec_plan_item: Union[tuple, List[Any], None],
        ) -> ExecutableNode:
            # Handle single tuple nodes (sequential)
            if isinstance(exec_plan_item, tuple):
                parent_node, node_name = exec_plan_item
                return ExecutableNode(
                    node_name=node_name,
                    node_list=[node_name],
                    task_list=[self.nodes[node_name].action],
                    execution_type="sequential",
                    interrupt=self.nodes[node_name].interrupt,
                )
            # Handle lists (parallel or sequential groups)
            elif isinstance(exec_plan_item, list):
                tasks = []
                node_names = []
                parent_names = []

                # First pass to collect all items
                for item in exec_plan_item:
                    if isinstance(item, tuple):
                        parent_node, node_name = item
                        tasks.append(self.nodes[node_name].action)
                        node_names.append(node_name)
                        parent_names.append(parent_node)
                    elif isinstance(item, list):
                        # Recursively create executable nodes for nested groups
                        executable_node = create_executable_node(item)
                        tasks.append(executable_node)
                        node_names.append(executable_node.node_list)
                    else:
                        raise ValueError(f"Expected tuple or list, got {type(item)}")

                # Check if all items have the same parent
                if all(parent == parent_names[0] for parent in parent_names):
                    execution_type = "parallel"
                else:
                    execution_type = "sequential"

                # Join all node names in the task list for the group name
                def get_node_name(name):
                    if isinstance(name, list):
                        # Flatten nested lists and handle each element
                        flattened = []
                        for item in name:
                            if isinstance(item, str):
                                flattened.append(item)
                            elif isinstance(item, tuple):
                                flattened.append(item[1])
                        return f"({'_'.join(flattened)})"
                    elif isinstance(name, tuple):
                        return name[1]
                    return name

                group_name = "_".join([get_node_name(name) for name in node_names])

                return ExecutableNode(
                    node_name=f"group_{group_name}",
                    task_list=tasks,
                    node_list=node_names,
                    execution_type=execution_type,
                    interrupt=None,
                )
            else:
                raise ValueError(f"Expected tuple or list, got {type(exec_plan_item)}")

        if not self.detailed_execution_path or any(
            item is None for item in self.detailed_execution_path
        ):
            raise ValueError(
                "No execution plan found. Please set detailed_execution_path."
            )

        self.execution_plan = [
            create_executable_node(item) for item in self.detailed_execution_path
        ]

        return self.execution_plan

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
        logger.debug(f"Chain status updated to: {status}")

    @internal_only
    def _save_state_and_buffers(self, node_name: str):
        if self.state:
            self._update_state_from_buffers()
            if self.checkpoint_storage:
                self.checkpoint_storage.save_checkpoint(self.state, self.chain_id)
        logger.debug(f"State updated after node: {node_name}")

    @internal_only
    def _get_interrupt_status(
        self, node_name: str
    ) -> Union[Literal["before", "after"], None]:
        return self.nodes[node_name].interrupt

    @internal_only
    def _execute(
        self, start_from: Optional[str] = None, timeout: Union[int, float] = 60 * 5
    ) -> None:
        """Execute the graph with concurrent and sequential execution based on the execution plan.

        Args:
            start_from: node name to start execution from
            timeout: maximum execution time in seconds
        """

        def execute_task(task: Callable, node_name: str) -> Any:
            """Execute a single task with proper state handling."""
            logger.debug(f"Executing task in node: {node_name}")

            # added this way to have access to class .self
            def run_task():
                result = task(state=self.state) if self._has_state else task()
                if result and self._has_state:
                    for state_field_name, state_field_value in result.items():
                        self.buffers[state_field_name].update(
                            state_field_value, node_name
                        )
                return result

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task)
                try:
                    result = future.result(timeout=timeout)
                    self._save_state_and_buffers(node_name)
                    self.executed_nodes.add(node_name)

                    return result
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    logger.error(f"Timeout in node {node_name}")
                    raise TimeoutError(f"Execution timeout in node {node_name}")

        def add_item_to_obj_store(obj_store: Union[List, Tuple], item):
            if isinstance(obj_store, list):
                obj_store.append(item)
                return obj_store  # Return the modified list
            elif isinstance(obj_store, tuple):
                return obj_store + (item,)  # Already returns the new tuple
            else:
                raise ValueError(f"Unsupported object store type: {type(obj_store)}")

        def extract_tasks_from_node(node, tasks=[]):
            """
            Extracts tasks from a node, including nested nodes
            returns lists for sequential execution and tuples for parallel execution
            """
            tasks = [] if node.execution_type == "sequential" else tuple()
            for task in node.task_list:
                if isinstance(task, ExecutableNode):
                    if task.execution_type == "sequential":
                        tasks = add_item_to_obj_store(
                            tasks, extract_tasks_from_node(task, [])
                        )
                    else:
                        tasks = add_item_to_obj_store(
                            tasks, extract_tasks_from_node(task, tuple())
                        )
                else:
                    tasks = add_item_to_obj_store(tasks, task)

            return tasks

        def execute_node(node: ExecutableNode, node_index: int) -> None:
            """Execute a single node or group of nodes with proper concurrency handling."""

            def execute_tasks(tasks, node_index: int):
                """Recursively execute tasks respecting list (sequential) and tuple (parallel) structures"""
                if isinstance(tasks, (list, tuple)):
                    if isinstance(tasks, list):
                        # Sequential execution
                        for task in tasks:
                            # Check chain status before continuing
                            if self.chain_status != ChainStatus.RUNNING:
                                return
                            execute_tasks(task, node_index)
                    else:
                        # Parallel execution using concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            futures = []
                            for task in tasks:
                                # Check chain status before submitting new tasks
                                if self.chain_status != ChainStatus.RUNNING:
                                    return
                                futures.append(
                                    executor.submit(execute_tasks, task, node_index)
                                )

                            # Wait for all futures to complete
                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    future.result(timeout=timeout)
                                except Exception as e:
                                    # Cancel remaining futures
                                    for f in futures:
                                        f.cancel()
                                    raise e
                else:
                    # Base case: execute individual task
                    node_name = getattr(tasks, "__name__", str(tasks))

                    # Skip if node was already executed
                    if node_name in self.executed_nodes:
                        return

                    # Handle before interrupts
                    if not isinstance(node_name, list):
                        if self._get_interrupt_status(node_name) == "before":
                            if not self.start_from:
                                self._save_state_and_buffers(node_name)
                                self.next_execution_node = node_name
                                self._update_chain_status(ChainStatus.PAUSE)
                                return

                    # Skip nodes until we reach start_from
                    if self.start_from and self.start_from != node_name:
                        return

                    # Cleaning up once start_from is reached
                    if self.start_from == node_name:
                        self.start_from = None
                        self._update_chain_status(ChainStatus.RUNNING)

                    try:
                        # Execute the task
                        result = execute_task(tasks, node_name)
                        self.last_executed_node = node_name

                        # Handle after interrupts
                        if not isinstance(node_name, list):
                            if self._get_interrupt_status(node_name) == "after":
                                if not self.start_from:
                                    self.next_execution_node = self.execution_plan[
                                        node_index + 1
                                    ].node_name
                                    self._update_chain_status(ChainStatus.PAUSE)
                                    return
                                else:
                                    self.start_from = None
                                    self._update_chain_status(ChainStatus.RUNNING)

                        return result

                    except concurrent.futures.TimeoutError:
                        logger.error(f"Timeout in node {node_name}")
                        raise TimeoutError(f"Execution timeout in node {node_name}")
                    except Exception as e:
                        logger.error(f"Error in node {node_name}: {str(e)}")
                        raise RuntimeError(
                            f"Error in node {node_name}: {str(e)}"
                        ) from e

            # Extract and execute tasks
            tasks = extract_tasks_from_node(node)
            execute_tasks(tasks, node_index)

        self._convert_execution_plan()
        self._update_chain_status(ChainStatus.RUNNING)
        self.start_from = start_from
        for node_index, node in enumerate(self.execution_plan):
            if self.chain_status == ChainStatus.RUNNING:
                execute_node(node, node_index)
            else:
                return

    def execute(
        self, start_from: Optional[str] = None, timeout: Union[int, float, None] = None
    ):
        if start_from is None:
            self.executed_nodes.clear()
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
