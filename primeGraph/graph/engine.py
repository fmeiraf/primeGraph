import asyncio
import copy
import logging
from typing import Union

from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.types import ChainStatus

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ExecutionFrame:
    def __init__(self, node_id, state):
        """
        A frame represents a branch of execution.

        :param node_id: The current node id to execute.
        :param state: The state (a dict) carried along this branch.
        """
        self.node_id = node_id
        self.state = state
        # Add branch tracking
        self.branch_id = None  # Will be set when branch is created
        self.target_convergence = None  # The convergence node this branch is heading towards


class GraphExecutor:
    def __init__(self, graph: Graph, timeout: int = 300):
        self.graph = graph
        self._resume_event = asyncio.Event()
        self._resume_event.clear()
        # Frames that represent pending execution branches.
        self.execution_frames = []

        self._visited_nodes = set()
        # Track branches and convergence
        self._branch_counter = 0
        self._active_branches = {}  # {convergence_node: set(branch_ids)}
        self._convergence_points = self._identify_convergence_points()
        self._timeout = timeout

    def _identify_convergence_points(self):
        """
        Identifies nodes that have multiple incoming edges (convergence points).
        Returns a dict of {node_id: number_of_incoming_edges}.
        """
        incoming_edges = {}
        for source, targets in self.graph.edges_map.items():
            for target in targets:
                incoming_edges[target] = incoming_edges.get(target, 0) + 1
        return {node: count for node, count in incoming_edges.items() if count > 1}

    def _find_next_convergence_point(self, start_node):
        """
        BFS to find the next convergence point this path leads to.
        Returns the convergence point node_id or None if path doesn't lead to one.
        """
        visited = set()
        queue = [(start_node, [])]

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

    async def _wait_for_resume(self):
        """
        When an interrupt occurs, wait until the user calls resume().
        """
        logger.debug("Execution paused. Waiting for resume...")
        await self._resume_event.wait()
        logger.debug("Execution resumed.")

    def resume(self):
        """
        Called by the user to resume execution after an interrupt.
        (In a real system you might trigger this from an external event.)
        """
        self._resume_event.set()
        self._resume_event.clear()

    async def execute(self):
        """
        Begin executing the graph from the START node.

        :param initial_state: Optional initial state (a dict) passed into the START node.
        """
        initial_state = self.graph.state
        if not initial_state:
            initial_state = {}
        initial_frame = ExecutionFrame(START, initial_state)
        self.execution_frames.append(initial_frame)
        await self._execute_all()

    async def _execute_all(self):
        """
        Process all pending execution frames.
        """
        while self.execution_frames and self.graph.chain_status != ChainStatus.DONE:
            frame = self.execution_frames.pop(0)
            await self._execute_frame(frame)

    def _node_is_executable(self, node_id: str) -> bool:
        """
        Check if a node is executable.
        """
        return node_id != START and node_id not in self._visited_nodes

    async def _execute_frame(self, frame: ExecutionFrame):
        """
        Process one execution branch. This will loop sequentially until a branch
        forks (parallel execution) or ends.
        """
        while True:
            if self.graph.chain_status != ChainStatus.RUNNING:
                self.graph._update_chain_status(ChainStatus.RUNNING)

            node_id = frame.node_id

            # Add debug logging for branch tracking
            if frame.branch_id is not None:
                logger.debug(f"Processing frame with branch_id {frame.branch_id} at node {node_id}")
                logger.debug(f"Current active branches: {self._active_branches}")

            if node_id == END:
                # Clean up branch tracking if this branch was being tracked
                if frame.branch_id is not None and frame.target_convergence:
                    if frame.branch_id in self._active_branches.get(frame.target_convergence, set()):
                        self._active_branches[frame.target_convergence].remove(frame.branch_id)

                # Mark execution as complete and return
                self.graph._update_chain_status(ChainStatus.DONE)
                # Clear any remaining execution frames to stop further processing
                self.execution_frames.clear()
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
                        return
                    else:
                        # Clear the active branches for this convergence point
                        self._active_branches[node_id].clear()

            if node_id not in self.graph.nodes:
                raise Exception(f"Node '{node_id}' not found in graph.")

            node = self.graph.nodes[node_id]

            # --- INTERRUPT BEFORE NODE EXECUTION ---
            if node.interrupt == "before":
                logger.debug(f"[Interrupt-before] About to execute node '{node_id}'.")
                self.graph._update_chain_status(ChainStatus.PAUSE)
                await self._wait_for_resume()

            # --- NODE EXECUTION ---
            logger.debug(f"Executing node '{node_id}' with state: {frame.state}")

            result = None
            if self._node_is_executable(node_id):
                try:
                    # Wrap execution in asyncio.wait_for for timeout
                    if node.is_async:
                        result = await asyncio.wait_for(
                            node.action(frame.state),
                            timeout=self._timeout,
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(node.action, frame.state),
                            timeout=self._timeout,
                        )
                except asyncio.TimeoutError:
                    logger.error(f"Node '{node_id}' execution timed out after {self._timeout} seconds")
                    self.graph._update_chain_status(ChainStatus.FAILED)
                    return
                except Exception as e:
                    logger.error(f"Error executing node '{node_id}': {str(e)}")
                    self.graph._update_chain_status(ChainStatus.FAILED)
                    return

                # check return values
                if not isinstance(result, Union[str, dict]):
                    raise ValueError(
                        f"Node '{node_id}' returned invalid result: {result}. Should return a string or a dict (router nodes)."
                    )

                # update buffers, state and checkpoints
                if isinstance(result, dict):
                    for state_field_name, state_field_value in result.items():
                        self.graph.buffers[state_field_name].update(state_field_value, node_id)

                    # update state
                    self.graph._update_state_from_buffers()

            # Mark node as visited
            self._visited_nodes.add(node_id)

            # Get next node before saving checkpoint
            children = self.graph.edges_map.get(node_id, [])
            next_node = children[0] if len(children) == 1 else None

            # If next node is a convergence point, handle branch cleanup before checkpoint
            if next_node and next_node in self._convergence_points:
                if frame.branch_id is not None and frame.target_convergence in self._active_branches:
                    self._active_branches[frame.target_convergence].remove(frame.branch_id)

            # Save checkpoint after branch cleanup
            if isinstance(result, dict):
                self.graph._save_checkpoint(node_id, self.get_full_state())

            # --- INTERRUPT AFTER NODE EXECUTION ---
            if node.interrupt == "after":
                logger.debug(f"[Interrupt-after] Executed node '{node_id}'.")
                self.graph._update_chain_status(ChainStatus.PAUSE)
                await self._wait_for_resume()

            # --- ROUTER NODES (OPTIONAL EXTENSION) ---
            if node.is_router:
                logger.debug(f"Router node '{node_id}' selecting branch based on result: {result}")

            # --- DETERMINING NEXT NODES ---
            if next_node:
                frame.node_id = next_node
                continue
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
                        self._active_branches[convergence_point].add(self._branch_counter)
                        self._branch_counter += 1
                    child_frames.append(child_frame)

                await asyncio.gather(*(self._execute_frame(child_frame) for child_frame in child_frames))
                return

    def get_full_state(self):
        """
        Returns a complete serializable snapshot of the executor's state.
        This includes all information needed to resume execution from a pause.

        :return: Dict containing all necessary state information
        """
        # Clean up any empty or completed branch sets before saving
        active_branches = {}
        for conv_point, branch_ids in self._active_branches.items():
            # Only include convergence points that:
            # 1. Haven't been visited yet
            # 2. Still have active branches
            # 3. Are actual convergence points
            if conv_point not in self._visited_nodes and branch_ids and conv_point in self._convergence_points:
                active_branches[conv_point] = list(branch_ids)

        return {
            "execution_frames": self.execution_frames.copy(),
            "visited_nodes": self._visited_nodes.copy(),
            "branch_counter": self._branch_counter,
            "active_branches": {k: v.copy() for k, v in self._active_branches.items()},
            "graph_state": self.graph.state,
            "chain_status": self.graph.chain_status.value,
        }

    def load_full_state(self, saved_state):
        """
        Restore the complete executor state from a saved snapshot.
        """
        # First restore visited nodes as it affects execution logic
        self._visited_nodes = set(saved_state["visited_nodes"])

        # Restore active branches exactly as they were
        self._active_branches = {}
        for conv_point, branch_ids in saved_state["active_branches"].items():
            branch_ids_set = set(branch_ids) if isinstance(branch_ids, list) else branch_ids
            if branch_ids_set:  # Only restore if there are active branches
                self._active_branches[conv_point] = branch_ids_set

        # Set branch counter to the saved value
        self._branch_counter = saved_state["branch_counter"]

        # Find the next nodes to execute based on visited nodes
        next_nodes = set()
        for node_id in self._visited_nodes:
            children = self.graph.edges_map.get(node_id, [])
            for child in children:
                if child not in self._visited_nodes and child != END:
                    # Check if this is a convergence point
                    if child in self._convergence_points:
                        # Only add convergence point if all its dependencies are visited
                        parents = self._get_node_parents(child)
                        if all(parent in self._visited_nodes for parent in parents):
                            next_nodes.add(child)
                    else:
                        next_nodes.add(child)

        # Create execution frames for the next nodes
        self.execution_frames = []
        for next_node in next_nodes:
            frame = ExecutionFrame(next_node, self.graph.state)
            # Find the convergence point this node is heading towards
            conv_point = self._find_next_convergence_point(next_node)
            if conv_point and conv_point in self._active_branches:
                # Only assign branch ID if the convergence point still has active branches
                active_branches = self._active_branches[conv_point]
                if active_branches:
                    frame.branch_id = next(iter(active_branches))  # Get first available branch ID
                    frame.target_convergence = conv_point
            self.execution_frames.append(frame)

        # Restore graph state
        self.graph.state = saved_state["graph_state"]
        self.graph._update_chain_status(ChainStatus(saved_state["chain_status"]))

    def _get_node_parents(self, node_id: str) -> set:
        """Get all parent nodes that have edges leading to the given node."""
        parents = set()
        for source, targets in self.graph.edges_map.items():
            if node_id in targets:
                parents.add(source)
        return parents

    def _get_branch_path_nodes(self, branch_id: int) -> set:
        """Helper method to get all nodes in a branch's path."""
        # This is a simplified version - you might need to adjust based on your graph structure
        path_nodes = set()
        for node_id in self.graph.nodes:
            if node_id not in {START, END} and node_id not in self._visited_nodes:
                path_nodes.add(node_id)
        return path_nodes
