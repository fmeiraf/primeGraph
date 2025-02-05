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


class GraphExecutor:
    def __init__(self, graph: Graph):
        self.graph = graph
        self._resume_event = asyncio.Event()
        self._resume_event.clear()
        # Frames that represent pending execution branches.
        self.execution_frames = []

        self._visited_nodes = set()

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
        return node_id != START and node_id != END and node_id not in self._visited_nodes

    async def _execute_frame(self, frame: ExecutionFrame):
        """
        Process one execution branch. This will loop sequentially until a branch
        forks (parallel execution) or ends.
        """
        while True:
            node_id = frame.node_id

            if node_id == END:
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

            # Optionally, merge the returned dict (if any) into state.
            # if isinstance(result, dict):
            #     frame.state.update(result)
            # logger.debug(f"Finished node '{node_id}'. Updated state: {frame.state}")

            # --- INTERRUPT AFTER NODE EXECUTION ---
            if node.interrupt == "after":
                logger.debug(f"[Interrupt-after] Executed node '{node_id}'.")
                await self._wait_for_resume()

            # --- ROUTER NODES (OPTIONAL EXTENSION) ---
            if node.is_router:
                # As an example, assume the node returns a key used to select the branch.
                # You would need to implement add_router_edge and store conditions.
                # For now, we assume the first child is always taken.
                logger.debug(f"Router node '{node_id}' selecting branch based on result: {result}")

            # --- DETERMINING NEXT NODES ---
            children = self.graph.edges_map.get(node_id, [])
            if not children:
                # No outgoing edge: branch is complete.
                logger.debug(f"Node '{node_id}' has no children. Ending branch.")
                return
            elif len(children) == 1:
                # Only one child: continue sequentially.
                frame.node_id = children[0]
                # Loop again with the updated frame.
                continue
            else:
                # Multiple children: launch parallel execution.
                logger.debug(f"Node '{node_id}' launches parallel branches: {children}")
                # Each child gets its own copy of the state.
                child_frames = [ExecutionFrame(child, copy.deepcopy(frame.state)) for child in children]
                # Execute all parallel branches concurrently.
                await asyncio.gather(*(self._execute_frame(child_frame) for child_frame in child_frames))
                # Once parallel branches finish, this branch is complete.
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
