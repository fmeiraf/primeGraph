import asyncio
import copy
import logging

from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph

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
    def __init__(self, graph: Graph):
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
        while self.execution_frames:
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

        Loop works as frame.id is updated and the while loop continues.
        node_id = node name that is the key on the edges_map dict

        edges_map = {
            "node_id": ["node_id1", "node_id2", "node_id3"],
            "node_id1": ["node_id4", "node_id5", "node_id6"],
            "node_id2": ["node_id7", "node_id8", "node_id9"],
            "node_id3": ["node_id10", "node_id11", "node_id12"],
        }

        """
        while True:
            node_id = frame.node_id

            if node_id == END:
                # Clean up branch tracking if this branch was being tracked
                if frame.branch_id is not None and frame.target_convergence:
                    self._active_branches[frame.target_convergence].remove(frame.branch_id)
                return

            # Check if this is a convergence point
            if node_id in self._convergence_points:
                if frame.branch_id is not None:
                    # Remove this branch from tracking
                    self._active_branches[node_id].remove(frame.branch_id)

                # Wait if there are still active branches targeting this convergence point
                if node_id in self._active_branches and self._active_branches[node_id]:
                    logger.debug(
                        f"Waiting at convergence point '{node_id}'. "
                        f"Still waiting for {len(self._active_branches[node_id])} branches"
                    )
                    return

            if node_id not in self.graph.nodes:
                raise Exception(f"Node '{node_id}' not found in graph.")

            node = self.graph.nodes[node_id]

            # --- INTERRUPT BEFORE NODE EXECUTION ---
            if node.interrupt == "before":
                logger.debug(f"[Interrupt-before] About to execute node '{node_id}'.")
                await self._wait_for_resume()

            # --- NODE EXECUTION ---
            logger.debug(f"Executing node '{node_id}' with state: {frame.state}")

            if self._node_is_executable(node_id):
                result = None
                if node.is_async:
                    result = await node.action(frame.state)
                else:
                    result = await asyncio.to_thread(node.action, frame.state)

            self._visited_nodes.add(node_id)

            # TODO: --- save buffers, update state and checkpoints ---
            # ...

            # --- INTERRUPT AFTER NODE EXECUTION ---
            if node.interrupt == "after":
                logger.debug(f"[Interrupt-after] Executed node '{node_id}'.")
                await self._wait_for_resume()

            # TODO: implement router nodes
            # --- ROUTER NODES (OPTIONAL EXTENSION) ---
            if node.is_router:
                # As an example, assume the node returns a key used to select the branch.
                # You would need to implement add_router_edge and store conditions.
                # For now, we assume the first child is always taken.
                logger.debug(f"Router node '{node_id}' selecting branch based on result: {result}")

            # --- DETERMINING NEXT NODES ---
            children = self.graph.edges_map.get(node_id, [])
            if not children:
                # Clean up branch tracking if this branch was being tracked
                if frame.branch_id is not None and frame.target_convergence:
                    self._active_branches[frame.target_convergence].remove(frame.branch_id)
                return
            elif len(children) == 1:
                frame.node_id = children[0]
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

    def get_state(self):
        """
        Returns a serializable snapshot of the pending execution frames.
        (E.g. to save to a database for later resumption.)
        """
        state = []
        for frame in self.execution_frames:
            state.append({"node_id": frame.node_id, "state": frame.state})
        return state

    def load_state(self, saved_state):
        """
        Restore pending execution frames from a saved state.
        :param saved_state: A list of dicts as produced by get_state().
        """
        self.execution_frames = []
        for frame_data in saved_state:
            self.execution_frames.append(ExecutionFrame(frame_data["node_id"], frame_data["state"]))
