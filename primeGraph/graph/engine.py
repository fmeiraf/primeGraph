import asyncio
import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from primeGraph.constants import END, START
from primeGraph.models.state import GraphState
from primeGraph.types import ChainStatus

if TYPE_CHECKING:
    from primeGraph.graph.base import Node
    from primeGraph.graph.executable import Graph
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ExecutionFrame:
    def __init__(self, node_id: str, state: GraphState | None = None):
        """
        A frame represents a branch of execution.

        :param node_id: The current node id to execute.
        :param state: The state (a dict) carried along this branch.
        """
        # Generate a unique ID for this execution frame
        self.frame_id = id(self)
        self.node_id = node_id
        self.state = state
        # Reference to the current node object (set in _execute_frame)
        self.current_node: Optional[Node] = None
        # Branch tracking for parallel execution
        self.branch_id: Optional[int] = None  # Will be set when branch is created
        self.target_convergence: Optional[str] = None  # The convergence node this branch is heading towards
        # When an interrupt is hit in a concurrent branch, pause by waiting on this event.
        self._pause_event: Optional[asyncio.Event] = None
        # If the branch is waiting at a convergence barrier, set it here.
        self.convergence_barrier: Optional[ConvergenceBarrier] = None
        # Flag to indicate the interrupt for this frame has been handled (after it was resumed)
        self.resumed = False

    def __repr__(self) -> str:
        return_value = f"ExecutionFrame(frame_id={self.frame_id}, node_id={self.node_id}, "
        return_value += f"branch_id={self.branch_id}, target_convergence={self.target_convergence}), "
        return_value += f"resumed={self.resumed}"
        if self.convergence_barrier:
            return_value += f", convergence_barrier={self.convergence_barrier}"
        return return_value


class ConvergenceBarrier:
    """
    A barrier to synchronize parallel branches at a convergence point.
    Once the required number of branches have reached the barrier and all are not paused,
    the barrier is lifted.
    """

    def __init__(self, count: int):
        self.count = count
        self.arrived = 0
        self.event = asyncio.Event()

    async def wait(self) -> None:
        self.arrived += 1
        if self.arrived >= self.count:
            self.event.set()
        await self.event.wait()


class Engine:
    def __init__(self, graph: "Graph", timeout: int = 300, max_node_iterations: int = 100):
        self.graph = graph
        self._resume_event = asyncio.Event()
        self._resume_event.clear()
        # For non-parallel (or "serialized") frames, we still keep a queue.
        self.execution_frames: List[ExecutionFrame | None] = []  # Used when running serially, not for parallel tasks.
        # Track visited nodes and per-node count
        self._visited_nodes: Set[str] = set()
        self._node_execution_count: Dict[str, int] = {}
        self._max_node_iterations = max_node_iterations
        # Branch tracking variables:
        self._branch_counter = 0
        self._active_branches: Dict[str, Set[int]] = {}  # {convergence_node: set(branch_ids)}
        self._convergence_points = self._identify_convergence_points()
        self._timeout = timeout
        # For true concurrency, we collect interrupted frames (by branch_id)
        self._interrupted_frames: Dict[int, ExecutionFrame | None] = {}  # {branch_id: ExecutionFrame}
        self._has_executed = False  # Flag to indicate if execute() has been called

        self._interrupt_triggered = False  # new flag

    def _identify_convergence_points(self) -> Dict[str, int]:
        """
        Identifies nodes that have multiple incoming edges (convergence points).
        Returns a dict of {node_id: number_of_incoming_edges}.
        """
        incoming_edges: Dict[str, int] = {}
        for source, targets in self.graph.edges_map.items():
            for target in targets:
                incoming_edges[target] = incoming_edges.get(target, 0) + 1
        return {node: count for node, count in incoming_edges.items() if count > 1}

    def _find_next_convergence_point(self, start_node: str) -> Optional[str]:
        """
        BFS to find the next convergence point this path leads to.
        Returns the convergence point node_id or None if path doesn't lead to one.
        """
        visited: Set[str] = set()
        queue: List[Tuple[str, List[str]]] = [(start_node, [])]

        while queue:
            current, path = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            # If we found a convergence point, return it
            if current in self._convergence_points:
                return current

            # Add children to queue
            children = self.graph.edges_map.get(current, [])
            for child in children:
                if child not in visited and child != END:
                    queue.append((child, path + [child]))

        return None

    async def _wait_for_resume(self) -> None:
        """
        When an interrupt occurs, wait until the user calls resume().
        """
        logger.debug("Execution paused. Waiting for resume...")
        await self._resume_event.wait()
        self._resume_event.clear()  # Clear the event immediately after it's triggered
        logger.debug("Execution resumed.")

    async def resume(self) -> None:
        """
        Resume execution: here we requeue all the paused frames (from a prior interrupt)
        and then continue processing.
        """

        logger.debug("Resuming execution on all paused branches...")
        paused_frames = [frame for frame in self._interrupted_frames.values() if frame is not None]
        for frame in paused_frames:
            frame._pause_event = None  # Clear the pause event
            frame.resumed = True  # Mark this frame as resumed so that its interrupt won't fire again
        self._interrupted_frames.clear()
        # Only extend execution frames with paused frames that aren't already there
        new_frames = [frame for frame in paused_frames if frame not in self.execution_frames]
        if new_frames:
            self.execution_frames.extend(new_frames)
            logger.debug(f"Extended execution frames: {self.execution_frames}")
            logger.debug(f"Paused frames: {paused_frames}")
            logger.debug(f"New frames: {new_frames}")

        self.graph._update_chain_status(ChainStatus.RUNNING)
        await self._execute_all()

    async def execute(self) -> None:
        """
        Begin executing the graph from the START node.
        This method will return either when the graph fully completes or when an interrupt occurs.
        """
        self._has_executed = True  # Mark that execute() has been run
        initial_state = self.graph.state
        # if not initial_state:
        #     initial_state = {}
        initial_frame = ExecutionFrame(START, initial_state)
        self.execution_frames.append(initial_frame)
        self.graph._update_chain_status(ChainStatus.RUNNING)
        await self._execute_all()

    async def _execute_all(self) -> None:
        """
        Process all pending execution frames.
        Now we check for chain_status so that if an interrupt has set it to PAUSE,
        the loop stops and execute() returns.
        """
        while self.execution_frames and self.graph.chain_status == ChainStatus.RUNNING:
            if len(self.execution_frames) > 1:
                logger.debug(f"Execution frames: {self.execution_frames}")
                await asyncio.gather(
                    *(self._execute_frame(frame) for frame in self.execution_frames if frame is not None)
                )
            else:
                frame = self.execution_frames.pop(0)
                if frame is not None:
                    await self._execute_frame(frame)

    def _node_is_executable(self, node_id: str) -> bool:
        """
        Check if a node is executable.
        """
        return node_id != START and node_id not in self._visited_nodes

    async def _execute_frame(self, frame: ExecutionFrame) -> None:
        """
        Execute a single frame. For branches created concurrently this method may run in an independent task.
        The key change here is that upon hitting an interrupt we do not wait on a pause event;
        we simply update the chain status to PAUSE and return.
        """
        while True:
            logger.debug(f"Executing frame {frame.frame_id}")
            if self.graph.chain_status != ChainStatus.RUNNING:
                self.graph._update_chain_status(ChainStatus.RUNNING)

            node_id = frame.node_id

            # Set current_node on the frame
            frame.current_node = self.graph.nodes[node_id] if node_id in self.graph.nodes else None

            # Check for cycle limits
            self._node_execution_count[node_id] = self._node_execution_count.get(node_id, 0) + 1
            if self._node_execution_count[node_id] > self._max_node_iterations:
                logger.error(
                    f"Node '{node_id}' has been executed {self._node_execution_count[node_id]} times, "
                    f"exceeding the maximum limit of {self._max_node_iterations}. "
                    "Possible infinite loop detected."
                )
                self.graph._update_chain_status(ChainStatus.FAILED)
                return

            # Add debug logging for branch tracking
            if frame.branch_id is not None:
                logger.debug(f"Processing frame with branch_id {frame.branch_id} at node {node_id}")
                logger.debug(f"Current active branches: {self._active_branches}")

            # Debug logging for all executions
            logger.debug(f"Executing node '{node_id}' with node object: {frame.current_node}")

            if node_id == END:
                # Clean up branch tracking if this branch was being tracked
                if frame.branch_id is not None and frame.target_convergence:
                    if frame.branch_id in self._active_branches.get(frame.target_convergence, set()):
                        self._active_branches[frame.target_convergence].remove(frame.branch_id)

                # Mark execution as complete and return
                self.graph._update_chain_status(ChainStatus.DONE)
                # Clear any remaining execution frames to stop further processing
                self.execution_frames.clear()
                self.graph._save_checkpoint(node_id, self.get_full_state())
                return

            # Check if this is a convergence point
            if node_id in self._convergence_points:
                if frame.branch_id is not None:
                    # Remove this branch from tracking only if it exists
                    if (
                        frame.target_convergence in self._active_branches
                        and frame.branch_id in self._active_branches[frame.target_convergence]
                    ):
                        logger.debug(f"Removing branch {frame.branch_id} from {frame.target_convergence}")
                        self._active_branches[frame.target_convergence].remove(frame.branch_id)
                        logger.debug(f"Active branches after removal: {self._active_branches}")

                # Check if we should wait at this convergence point
                # Only wait if there are unvisited branches that need to converge
                if node_id in self._active_branches:
                    unvisited_branches = set()
                    for branch_id in self._active_branches[node_id]:
                        # Check if this branch's path hasn't been visited yet
                        branch_path_nodes = self._get_branch_path_nodes(branch_id)
                        if not all(node in self._visited_nodes for node in branch_path_nodes):
                            unvisited_branches.add(branch_id)

                    if unvisited_branches:
                        logger.debug(
                            f"Waiting at convergence point '{node_id}'. "
                            f"Still waiting for branches: {unvisited_branches}"
                        )
                        # Instead of waiting here, simply return so that execute() stops.
                        return
                    else:
                        # Clear the active branches for this convergence point
                        self._active_branches[node_id].clear()

            if node_id not in self.graph.nodes:
                raise Exception(f"Node '{node_id}' not found in graph.")

            node = self.graph.nodes[node_id]

            # --- INTERRUPT BEFORE NODE EXECUTION ---
            if node.interrupt == "before" and node_id not in self._visited_nodes and not frame.resumed:
                if frame._pause_event is None:
                    frame._pause_event = asyncio.Event()
                    self._interrupted_frames[frame.branch_id or 0] = frame
                logger.debug(
                    f"[Interrupt-before] [[ {frame.node_id} ]] "
                    f"Branch {frame.branch_id} pausing before executing node '{node_id}'."
                )
                self.graph._update_chain_status(ChainStatus.PAUSE)
                self.graph._save_checkpoint(node_id, self.get_full_state())
                # Remove the frame from execution_frames to prevent reprocessing
                if frame in self.execution_frames:
                    self.execution_frames.remove(frame)
                return

            # --- NODE EXECUTION ---
            logger.debug(f"Executing node '{node_id}' with state: {frame.state}")

            result = None
            if self._node_is_executable(node_id):
                try:
                    import inspect
                    import types

                    # Get the node action and inspect its signature
                    node_action = node.action
                    sig = inspect.signature(node_action)
                    # If the action expects a parameter named "self", bind it to the node.
                    if "self" in sig.parameters:
                        node_action = types.MethodType(node_action, node)

                    # Now call the (possibly bound) action with state if available.
                    if node.is_async:
                        result = await asyncio.wait_for(
                            node_action(self.graph.state) if self.graph.state else node_action(),  # type: ignore[arg-type]
                            timeout=self._timeout,
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(node_action, self.graph.state)  # type: ignore[func-returns-value]
                            if self.graph.state
                            else asyncio.to_thread(node_action),
                            timeout=self._timeout,
                        )

                    # check return values
                    if not isinstance(result, (str, dict)):
                        raise ValueError(
                            f"Node '{node_id}' returned invalid result: {result}. Should return a string or a dict."
                        )

                    # update buffers, state and checkpoints
                    if isinstance(result, dict):
                        for state_field_name, state_field_value in result.items():
                            self.graph.buffers[state_field_name].update(state_field_value, node_id)
                        self.graph._update_state_from_buffers()

                    # After successful execution, clear interrupt state if this was the interrupted node
                    if self._interrupted_frames.get(frame.branch_id or 0) == frame:
                        self._interrupted_frames.pop(frame.branch_id or 0)

                except asyncio.TimeoutError as e:
                    logger.error(
                        f"Execution timeout: Node '{node_id}' execution timed out after {self._timeout} seconds"
                    )
                    self.graph._update_chain_status(ChainStatus.FAILED)
                    self.graph._save_checkpoint(node_id, self.get_full_state())
                    raise TimeoutError(
                        f"Execution timeout: Node '{node_id}' execution timed out after {self._timeout} seconds"
                    ) from e
                except Exception as e:
                    logger.error(f"Error executing node '{node_id}': {str(e)}")
                    self.graph._update_chain_status(ChainStatus.FAILED)
                    self.graph._save_checkpoint(node_id, self.get_full_state())
                    raise RuntimeError(f"Task failed: Error executing node '{node_id}': {str(e)}") from e

            self._visited_nodes.add(node_id)

            # *** NEW CHECK: If any branch triggered an interrupt, pause before scheduling more nodes.
            if self.graph.chain_status != ChainStatus.RUNNING:
                self._interrupted_frames[frame.branch_id or 0] = frame
                return

            # Get next node before saving checkpoint
            children = self.graph.edges_map.get(node_id, [])
            next_node = children[0] if len(children) == 1 else None

            # Log the children for debugging
            logger.debug(f"Node '{node_id}' has children: {children}")

            # If next node is a convergence point, handle branch cleanup before checkpoint
            if next_node and next_node in self._convergence_points:
                if frame.branch_id is not None and frame.target_convergence in self._active_branches:
                    self._active_branches[frame.target_convergence].remove(frame.branch_id)

            # Save checkpoint after branch cleanup
            if isinstance(result, dict):
                self.graph._save_checkpoint(node_id, self.get_full_state())

            # --- ROUTER NODES (OPTIONAL EXTENSION) ---
            if node.is_router:
                if not isinstance(result, str):
                    raise ValueError(
                        f"Router node '{node_id}' must return a string node_id. Got: {result} of type {type(result)}"
                    )
                if result not in self.graph.nodes:
                    raise ValueError(
                        f"Router node '{node_id}' returned invalid node_id: "
                        f"'{result}'. Must be a valid node in the graph."
                    )

                target_count = self._node_execution_count.get(result, 0)
                if target_count >= self._max_node_iterations:
                    logger.error(
                        f"Router target node '{result}' has been executed {target_count} times, "
                        f"exceeding the maximum limit of {self._max_node_iterations}. "
                        "Possible infinite loop detected."
                    )
                    self.graph._update_chain_status(ChainStatus.FAILED)
                    self.graph._save_checkpoint(node_id, self.get_full_state())
                    return

                logger.debug(f"Router node '{node_id}' routing to node: {result}")

                # Remove all downstream nodes from visited_nodes
                nodes_to_clean = self._get_downstream_nodes(result)
                for node_to_clean in nodes_to_clean:
                    if node_to_clean in self._visited_nodes:
                        logger.debug(f"Removing downstream node '{node_to_clean}' from visited nodes for re-execution")
                        self._visited_nodes.remove(node_to_clean)

                # Reset the frame's interrupt state before routing back.
                frame.node_id = result
                frame.resumed = False  # Reset resumed to ensure the interrupt check runs again.
                frame._pause_event = None  # Clear any previous pause event.
                continue

            # --- DETERMINING NEXT NODES ---
            children = self.graph.edges_map.get(node_id, [])
            if len(children) == 1:
                frame.node_id = children[0]
            else:
                logger.debug(f"Node '{node_id}' launches parallel branches: {children}")
                child_frames = []

                # Find convergence point for these new branches
                convergence_point = None
                for child in children:
                    conv = self._find_next_convergence_point(child)
                    if conv:
                        convergence_point = conv
                        break

                if convergence_point:
                    # Initialize tracking for this set of branches
                    self._active_branches.setdefault(convergence_point, set())

                # Create child frames with branch tracking
                for child in children:
                    child_frame = ExecutionFrame(child, copy.deepcopy(frame.state))
                    if convergence_point:
                        child_frame.branch_id = self._branch_counter
                        child_frame.target_convergence = convergence_point
                        child_frame.convergence_barrier = ConvergenceBarrier(count=len(children))
                        self._active_branches[convergence_point].add(self._branch_counter)
                        self._branch_counter += 1
                    child_frames.append(child_frame)

                self.execution_frames.extend(child_frames)

            # --- INTERRUPT AFTER NODE EXECUTION ---
            if node.interrupt == "after":
                logger.debug(f"[Interrupt-after] Executed node '{node_id}'.")
                self.graph._update_chain_status(ChainStatus.PAUSE)
                self._interrupted_frames[frame.branch_id or 0] = frame if len(children) == 1 else None
                self.graph._save_checkpoint(node_id, self.get_full_state())
                return

            if len(children) == 1:
                continue

            logger.debug(f"Laucing Parallel branches: {child_frames}")
            await asyncio.gather(*(self._execute_frame(child_frame) for child_frame in child_frames))
            return

    def get_full_state(self) -> Dict:
        """
        Returns a complete serializable snapshot of the executor's state.
        This includes all information needed to resume execution from a pause.

        :return: Dict containing all necessary state information
        """
        # Clean up any empty or completed branch sets before saving
        active_branches = {k: list(v) for k, v in self._active_branches.items() if v}

        state = {
            "execution_frames": self.execution_frames.copy(),
            "visited_nodes": self._visited_nodes.copy(),
            "node_execution_count": self._node_execution_count.copy(),  # Add execution count to state
            "branch_counter": self._branch_counter,
            "active_branches": active_branches,
            "chain_status": self.graph.chain_status.value,
            "interrupted_frames": self._interrupted_frames.copy(),
        }
        return state

    def load_full_state(self, saved_state: Dict[str, Any]) -> None:
        """
        Load the engine's internal execution state from a saved checkpoint.
        NOTE: This method *only* loads the engine's internal state (frames, visited nodes, etc.).
        The actual graph state (self.graph.state) should be loaded separately, typically
        before calling this method (e.g., in Graph.load_from_checkpoint).

        Args:
            saved_state: The state dictionary from a checkpoint, containing engine state.
        """
        # --- Load Engine-Specific State ---
        # Restore execution frames, visited nodes, etc. from saved_state
        # Ensure we handle potential missing keys gracefully

        # Restore execution frames
        # TODO: ExecutionFrame needs proper serialization/deserialization if it contains complex objects
        self.execution_frames = saved_state.get("execution_frames", [])

        # Restore visited nodes
        self._visited_nodes = set(saved_state.get("visited_nodes", set()))

        # Restore node execution count
        self._node_execution_count = saved_state.get("node_execution_count", {})

        # Restore branch counter
        self._branch_counter = saved_state.get("branch_counter", 0)

        # Restore active branches (convert list back to set)
        self._active_branches = {k: set(v) for k, v in saved_state.get("active_branches", {}).items()}

        # Restore interrupted frames
        # TODO: Need proper serialization/deserialization for interrupted frames
        self._interrupted_frames = saved_state.get("interrupted_frames", {})

        # --- DO NOT OVERWRITE self.graph.state ---
        # The graph state (self.graph.state) is assumed to be loaded
        # by the Graph subclass (e.g., ToolGraph.load_from_checkpoint)
        # using its specific deserialization logic (like Pydantic's model_validate_json).
        # We explicitly avoid overwriting it here.

        # --- Restore Chain Status (but don't rely on is_paused from engine state) ---
        # Get chain status from the saved engine state if present
        chain_status_value = saved_state.get("chain_status", ChainStatus.DONE.value)
        try:
            chain_status = ChainStatus(chain_status_value)
        except ValueError:
            print(
                f"[Engine.load_full_state] Warning: "
                f"Invalid chain_status value '{chain_status_value}', defaulting to DONE."
            )
            chain_status = ChainStatus.DONE

        # Update the graph's chain status based *only* on the engine state's record.
        # Do not infer based on is_paused, as that belongs to the GraphState.
        self.graph.chain_status = chain_status
        print(f"[Engine.load_full_state] Restored engine state. Graph chain_status set to: {chain_status}")

        # --- REMOVED ---
        # - Creation of `new_state`
        # - Loop setting attributes on `new_state`
        # - Assignment `self.graph.state = new_state`
        # - Logic relying on `is_paused` from `saved_state`
        # - Conversion logic for `paused_tool_result` (should be handled by GraphState deserialization)

    def _get_node_parents(self, node_id: str) -> Set[str]:
        """Get all parent nodes that have edges leading to the given node."""
        parents: Set[str] = set()
        for source, targets in self.graph.edges_map.items():
            if node_id in targets:
                parents.add(source)
        return parents

    def _get_branch_path_nodes(self, branch_id: int) -> Set[str]:
        """Helper method to get all nodes in a branch's path."""
        # This is a simplified version - you might need to adjust based on your graph structure
        path_nodes: Set[str] = set()
        for node_id in self.graph.nodes:
            if node_id not in {START, END} and node_id not in self._visited_nodes:
                path_nodes.add(node_id)
        return path_nodes

    def _get_downstream_nodes(self, start_node: str) -> Set[str]:
        """
        Get all downstream nodes from a starting node using BFS.
        This includes the start node itself.
        """
        downstream_nodes: Set[str] = set([start_node])
        queue: List[str] = [start_node]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)
            children = self.graph.edges_map.get(current, [])

            for child in children:
                if child != END:  # Don't include END node in downstream nodes
                    downstream_nodes.add(child)
                    if child not in visited:
                        queue.append(child)

        return downstream_nodes
