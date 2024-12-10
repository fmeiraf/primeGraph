import asyncio
import concurrent.futures
import inspect
import logging
import uuid
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
from tiny_graph.buffer.history import HistoryBuffer
from tiny_graph.checkpoint.base import StorageBackend
from tiny_graph.graph.base import BaseGraph
from tiny_graph.models.state import GraphState
from tiny_graph.types import ChainStatus
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


class Graph(BaseGraph):
    def __init__(
        self,
        state: Union[BaseModel, None] = None,
        checkpoint_storage: Optional[StorageBackend] = None,
        chain_id: Optional[str] = None,
        execution_timeout: Union[int, float] = 60 * 5,
    ):
        super().__init__(state)

        # State management
        self.initial_state = state
        self.state = state
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
        self.next_execution_node = None
        self.execution_timeout = execution_timeout
        self.start_from = None
        self.executed_nodes = set()

    def _assign_buffers(self):
        self.buffers = {
            field_name: BufferFactory.create_buffer(field_name, field_type)
            for field_name, field_type in self.state_schema.items()
        }

    def _reset_state(self, new_state: Union[BaseModel, None] = None):
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

    def _get_chain_status(self) -> ChainStatus:
        return self.chain_status

    def _clean_graph_variables(self, new_state: Union[BaseModel, None] = None):
        # One-off set up variables
        self.next_execution_node = None
        self.executed_nodes = set()
        self.start_from = None
        self.last_executed_node = None
        self.chain_status = ChainStatus.IDLE

        # Re-assign buffers and reset state
        if self.state_schema:
            self._assign_buffers()

        # Reset state to first assigned state (from graph init)
        self._reset_state(new_state)

    def _update_chain_status(self, status: ChainStatus):
        self.chain_status = status
        logger.debug(f"Chain status updated to: {status}")

    @internal_only
    def _update_state_from_buffers(self):
        for field_name, buffer in self.buffers.items():
            if buffer._ready_for_consumption:
                setattr(self.state, field_name, buffer.consume_last_value())

    @internal_only
    def _update_buffers_from_state(self):
        for field_name, buffer in self.buffers.items():
            if isinstance(buffer, HistoryBuffer):
                buffer.set_value(getattr(self.state, field_name))
            else:
                buffer.update(getattr(self.state, field_name), "update_from_state")

    def _save_checkpoint(self, node_name: str):
        if self.state:
            if self.checkpoint_storage:
                # TODO: add checkpoint_id or make it optional
                self.checkpoint_storage.save_checkpoint(
                    state_instance=self.state,
                    chain_id=self.chain_id,
                    chain_status=self.chain_status,
                    next_execution_node=self.next_execution_node,
                    executed_nodes=self.executed_nodes,
                )
        logger.debug(f"Checkpoint saved after node: {node_name}")

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

                # Handle router node results
                if self.nodes[node_name].is_router:
                    if not result or not isinstance(result, str):
                        raise ValueError(
                            f"Router node '{node_name}' must return a valid node name"
                        )

                    # Validate the returned route
                    if result not in self.nodes[node_name].possible_routes:
                        raise ValueError(
                            f"Router node '{node_name}' returned invalid route: {result}"
                        )

                    # Get the path for this route
                    router_paths = self.router_paths.get(node_name, {})
                    chosen_path = router_paths.get(result, [])

                    if chosen_path:
                        logger.debug(f"Router {node_name} chose path: {chosen_path}")
                        # Update execution plan to only include the chosen path
                        self._update_execution_plan(node_name, chosen_path)
                        # Set start_from to the first node in the chosen path
                        self.start_from = chosen_path[0]
                        logger.debug(
                            f"Updated execution path: {self.detailed_execution_path}"
                        )
                        logger.debug(f"Next node to execute: {self.start_from}")

                elif result and self._has_state:
                    for state_field_name, state_field_value in result.items():
                        self.buffers[state_field_name].update(
                            state_field_value, node_name
                        )
                return result

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task)
                try:
                    result = future.result(timeout=timeout)
                    self._update_state_from_buffers()
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
                            self._save_checkpoint(
                                self.execution_plan[node_index].node_name
                            )
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
                            self._save_checkpoint(
                                self.execution_plan[node_index].node_name
                            )
                else:
                    # Base case: execute individual task
                    node_name = next(
                        (
                            name
                            for name, node in self.nodes.items()
                            if node.action == tasks
                        ),
                        str(tasks),  # fallback to str(tasks) if not found
                    )

                    # Skip if node was already executed
                    if node_name in self.executed_nodes:
                        return

                    # Handle before interrupts
                    if not isinstance(node_name, list):
                        if self._get_interrupt_status(node_name) == "before":
                            if not self.start_from:
                                # self._save_state_and_buffers(node_name)
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

    @internal_only
    def execute(
        self,
        start_from: Optional[str] = None,
        timeout: Union[int, float] = 60 * 5,
    ):
        if start_from is None:
            self.executed_nodes.clear()
        if not timeout:
            timeout = self.execution_timeout

        self._execute(start_from, timeout)

    def resume(self, start_from: Optional[str] = None):
        if not self.next_execution_node and not start_from:
            logger.info(
                "resume method should either specify a start_from node or be part of a chain call (execute)"
            )
            raise ValueError(
                "resume method should either specify a start_from node or be part of a chain call (execute)"
            )

        # ensure that buffers are updated from state
        self._update_buffers_from_state()

        if start_from:
            self.start_from = start_from
            self.execute(start_from=start_from)
        else:
            self.execute(start_from=self.next_execution_node)
        return

    def start(
        self, chain_id: Optional[str] = None, timeout: Union[int, float] = None
    ) -> str:
        """Start a new graph execution with a new chain id"""
        if chain_id:
            self.chain_id = chain_id
        else:
            self.chain_id = f"chain_{uuid.uuid4()}"

        if self.chain_status != ChainStatus.IDLE:
            self._clean_graph_variables()
        self.execute(timeout=timeout)
        return self.chain_id

    def load_from_checkpoint(
        self, chain_id: str, checkpoint_id: Optional[str] = None
    ) -> None:
        """Load graph state and execution variables from a checkpoint.

        Args:
            checkpoint_id: Optional specific checkpoint ID to load. If None, loads the last checkpoint.
        """
        if not self.checkpoint_storage:
            raise ValueError(
                "Checkpoint storage must be configured to load from checkpoint"
            )

        # Get checkpoint ID if not specified
        if not checkpoint_id:
            checkpoint_id = self.checkpoint_storage.get_last_checkpoint_id(chain_id)
            if not checkpoint_id:
                raise ValueError(f"No checkpoints found for chain {chain_id}")

        # Load checkpoint data
        checkpoint = self.checkpoint_storage.load_checkpoint(
            state_instance=self.state,
            chain_id=chain_id,
            checkpoint_id=checkpoint_id,
        )

        # Verify state class matches
        current_state_class = (
            f"{self.state.__class__.__module__}.{self.state.__class__.__name__}"
        )
        if current_state_class != checkpoint.state_class:
            raise ValueError(
                f"State class mismatch. Current: {current_state_class}, "
                f"Checkpoint: {checkpoint.state_class}"
            )

        self._clean_graph_variables()
        # Update state from serialized data
        self.state = self.initial_state.__class__.model_validate_json(checkpoint.data)

        # Update buffers with current state values
        for field_name, buffer in self.buffers.items():
            buffer.set_value(getattr(self.state, field_name))

        # Update execution variables
        self.chain_id = checkpoint.chain_id
        self.chain_status = checkpoint.chain_status
        self.next_execution_node = checkpoint.next_execution_node
        if checkpoint.executed_nodes:
            self.executed_nodes = checkpoint.executed_nodes

        logger.debug(f"Loaded checkpoint {checkpoint_id} for chain {self.chain_id}")

    @internal_only
    async def _execute_async(
        self, start_from: Optional[str] = None, timeout: Union[int, float] = 60 * 5
    ) -> None:
        """Async version of execute method"""

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

        async def execute_task(task: Callable, node_name: str) -> Any:
            """Execute a single task with proper state handling."""
            logger.debug(f"Executing task in node: {node_name}")

            async def run_task():
                logger.debug(f"Running task: {task}, has state: {self._has_state}")
                if inspect.iscoroutinefunction(task):
                    # Handle async functions
                    result = (
                        await task(state=self.state)
                        if self._has_state
                        else await task()
                    )
                    # Ensure we're getting the actual result, not a coroutine
                    if inspect.iscoroutine(result):
                        result = await result
                else:
                    # Handle CPU-bound sync functions by running them in a thread
                    if self._has_state:
                        result = await asyncio.to_thread(task, state=self.state)
                    else:
                        result = await asyncio.to_thread(task)

                if result and self._has_state:
                    for state_field_name, state_field_value in result.items():
                        self.buffers[state_field_name].update(
                            state_field_value, node_name
                        )
                return result

            try:
                result = await asyncio.wait_for(run_task(), timeout=timeout)
                self._update_state_from_buffers()
                self.executed_nodes.add(node_name)
                return result
            except asyncio.TimeoutError:
                logger.error(f"Timeout in node {node_name}")
                raise TimeoutError(f"Execution timeout in node {node_name}")

        async def execute_tasks(tasks, node_index: int):
            """Recursively execute tasks respecting list (sequential) and tuple (parallel) structures"""
            if isinstance(tasks, (list, tuple)):
                if isinstance(tasks, list):
                    # Sequential execution
                    for task in tasks:
                        if self.chain_status != ChainStatus.RUNNING:
                            return
                        await execute_tasks(task, node_index)
                        self._save_checkpoint(self.execution_plan[node_index].node_name)
                else:
                    # Parallel execution using asyncio.gather
                    if self.chain_status != ChainStatus.RUNNING:
                        return

                    # Check if any task in the parallel group has a "before" interrupt
                    for task in tasks:
                        if callable(task):
                            node_name = getattr(
                                task,
                                "__node_name__",
                                getattr(task, "__name__", str(task)),
                            )
                            if self._get_interrupt_status(node_name) == "before":
                                if not self.start_from:
                                    self.next_execution_node = node_name
                                    self._update_chain_status(ChainStatus.PAUSE)
                                    return

                    # Create a list of coroutines for parallel execution
                    parallel_tasks = []
                    for task in tasks:
                        # If it's a callable (actual task)
                        if callable(task):
                            node_name = getattr(
                                task,
                                "__node_name__",
                                getattr(task, "__name__", str(task)),
                            )
                            if node_name not in self.executed_nodes:
                                # Skip nodes until we reach start_from
                                if self.start_from and self.start_from != node_name:
                                    continue

                                # Clean up start_from when reached
                                if self.start_from == node_name:
                                    self.start_from = None
                                    self._update_chain_status(ChainStatus.RUNNING)

                                parallel_tasks.append(execute_task(task, node_name))
                        # If it's a nested structure (list/tuple)
                        elif isinstance(task, (list, tuple)):
                            parallel_tasks.append(execute_tasks(task, node_index))

                    # Execute all tasks in parallel if we have any
                    if parallel_tasks:
                        await asyncio.gather(*parallel_tasks)

                        # Check if any task in the parallel group has an "after" interrupt
                        for task in tasks:
                            if callable(task):
                                node_name = getattr(
                                    task,
                                    "__node_name__",
                                    getattr(task, "__name__", str(task)),
                                )
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

                        self._save_checkpoint(self.execution_plan[node_index].node_name)
            else:
                # Base case: execute individual task
                node_name = next(
                    (name for name, node in self.nodes.items() if node.action == tasks),
                    str(tasks),  # fallback to str(tasks) if not found
                )

                # Skip if node was already executed
                if node_name in self.executed_nodes:
                    return

                # Handle before interrupts
                if not isinstance(node_name, list):
                    if self._get_interrupt_status(node_name) == "before":
                        if not self.start_from:
                            # self._save_state_and_buffers(node_name)
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
                    result = await execute_task(tasks, node_name)
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

                except asyncio.TimeoutError:
                    logger.error(f"Timeout in node {node_name}")
                    raise TimeoutError(f"Execution timeout in node {node_name}")
                except Exception as e:
                    logger.error(f"Error in node {node_name}: {str(e)}")
                    raise RuntimeError(f"Error in node {node_name}: {str(e)}") from e

        async def execute_node(node: ExecutableNode, node_index: int) -> None:
            """Execute a single node or group of nodes with proper concurrency handling."""
            tasks = extract_tasks_from_node(node)
            await execute_tasks(tasks, node_index)

        # Initialize execution
        self._convert_execution_plan()
        self._update_chain_status(ChainStatus.RUNNING)
        self.start_from = start_from

        # Execute nodes
        for node_index, node in enumerate(self.execution_plan):
            if self.chain_status == ChainStatus.RUNNING:
                await execute_node(node, node_index)
            else:
                return

    @internal_only
    async def execute_async(
        self,
        start_from: Optional[str] = None,
        timeout: Union[int, float] = 60 * 5,
    ):
        """Async version of execute method"""
        if start_from is None:
            self.executed_nodes.clear()
        if not timeout:
            timeout = self.execution_timeout

        await self._execute_async(start_from, timeout)

    async def start_async(
        self, chain_id: Optional[str] = None, timeout: Union[int, float] = None
    ) -> str:
        """Async version of start method"""
        if chain_id:
            self.chain_id = chain_id
        else:
            self.chain_id = f"chain_{uuid.uuid4()}"

        if self.chain_status != ChainStatus.IDLE:
            self._clean_graph_variables()
        await self.execute_async(timeout=timeout)
        return self.chain_id

    async def resume_async(self, start_from: Optional[str] = None):
        """Async version of resume method"""
        if not self.next_execution_node and not start_from:
            logger.info(
                "resume method should either specify a start_from node or be part of a chain call (execute)"
            )
            raise ValueError(
                "resume method should either specify a start_from node or be part of a chain call (execute)"
            )

        self._update_buffers_from_state()

        if start_from:
            self.start_from = start_from
            await self.execute_async(start_from=start_from)
        else:
            await self.execute_async(start_from=self.next_execution_node)

    def _update_execution_plan(self, router_node: str, chosen_node: str) -> None:
        """Update detailed execution plan to only include the chosen path from a router node."""

        def _find_node_in_nested(node_name: str, path: List[Any]) -> bool:
            """Helper function to check if a node exists in a nested structure."""
            for item in path:
                if isinstance(item, tuple) and item[1] == node_name:
                    return True
                elif isinstance(item, list):
                    if _find_node_in_nested(node_name, item):
                        return True
            return False

        def _find_node_index(node_name: str, path: List[Any]) -> int:
            """Find the index of a node in the detailed execution path."""
            for i, item in enumerate(path):
                # Check tuples (node entries)
                if isinstance(item, tuple) and item[1] == node_name:
                    return i
                # Check nested lists (parallel paths)
                elif isinstance(item, list):
                    if any(
                        _find_node_in_nested(node_name, [subitem]) for subitem in item
                    ):
                        return i
            raise ValueError(f"Node {node_name} not found in detailed execution path")

        def isolate_chosen_path(path: List[Any], chosen_node: str) -> List[Any]:
            """Find and isolate the path containing the chosen node."""
            if isinstance(path, list):
                for item in path:
                    if isinstance(item, list):
                        # For nested parallel paths
                        for subpath in item:
                            if isinstance(subpath, list):
                                result = isolate_chosen_path(subpath, chosen_node)
                                if result:
                                    return result
                            elif (
                                isinstance(subpath, tuple) and subpath[1] == chosen_node
                            ):
                                return subpath
                    elif isinstance(item, tuple) and item[1] == chosen_node:
                        return path
            return None

        # Find indices in the detailed execution path
        router_idx = _find_node_index(router_node, self.detailed_execution_path)

        # Find the parallel paths group that comes after the router
        if router_idx + 1 >= len(self.detailed_execution_path):
            return

        parallel_paths = self.detailed_execution_path[router_idx + 1]
        if not isinstance(parallel_paths, list):
            return

        # Find the specific path containing the chosen node
        chosen_path = None
        for path in parallel_paths:
            if _find_node_in_nested(chosen_node, [path]):
                chosen_path = path
                break

        if not chosen_path:
            raise ValueError(
                f"Chosen node {chosen_node} not found in any path after router"
            )

        # Update the detailed execution path to only include the chosen path
        self.detailed_execution_path[router_idx + 1] = [chosen_path]
