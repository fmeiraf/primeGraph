import ast
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Self,
    Set,
    Tuple,
    Union,
)

from pydantic import BaseModel

from tiny_graph.constants import END, START


class Edge(NamedTuple):
    start_node: str
    end_node: str


class Node(NamedTuple):
    name: str
    action: Callable[..., None]
    metadata: Optional[Dict[str, Any]] = None
    is_async: bool = False
    is_router: bool = False
    possible_routes: Optional[Set[str]] = None


class Graph:
    def __init__(self, state: Union[BaseModel, NamedTuple, None] = None):
        self.nodes: Dict[str, Node] = {}
        self.edges: Set[Edge] = set()
        self.is_compiled: bool = False
        self.tasks: List[Callable[..., None]] = []
        self.state: Union[BaseModel, NamedTuple, None] = state
        self.state_schema: Dict[str, type] = _get_schema(state)

    @property
    def _all_nodes(self) -> List[str]:
        return list(self.nodes.keys())

    @property
    def _all_edges(self) -> Set[Tuple[str, str]]:
        return self.edges

    def _get_return_values(self, func: Callable) -> Set[str]:
        """Extract all string literal return values from a function"""
        source = inspect.getsource(func)
        # Dedent the source code before parsing
        source = inspect.cleandoc(source)
        tree = ast.parse(source)

        return_values = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                # Handle direct string returns
                if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str
                ):
                    return_values.add(node.value.value)
                # Handle string variable returns
                elif isinstance(node.value, ast.Name):
                    # Note: This is a limitation - we can only detect direct string returns
                    # Variable returns would require more complex static analysis
                    pass
        return return_values

    def node(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Decorator to add a node to the graph

        Args:
            name: Optional name for the node. If None, uses the function name
            metadata: Optional metadata dictionary
        """

        def decorator(func: Callable[..., None]):
            if self.is_compiled:
                raise ValueError("Cannot add nodes after compiling the graph")

            # Checking for reserved node names
            node_name = name if name is not None else func.__name__
            if node_name in [START, END]:
                raise ValueError(f"Node name '{node_name}' is reserved")

            # Checking for async calls
            is_async = inspect.iscoroutinefunction(func)

            # Check if this is a router node by looking for return statements
            return_values = self._get_return_values(func)
            is_router = len(return_values) > 0

            self.nodes[node_name] = Node(
                node_name,
                func,
                metadata,
                is_async,
                is_router,
                return_values if is_router else None,
            )
            return func

        return decorator

    def add_edge(self, start_node: str, end_node: str) -> Self:
        self.edges.add(Edge(start_node, end_node))
        return self

    def validate(self) -> Self:
        """Validate that all nodes are on valid paths from start nodes.

        Raises:
            ValueError: If orphaned nodes are found or if there are cycles in the graph
        """
        # Find start nodes (nodes with no incoming edges)
        incoming_edges = {edge.end_node for edge in self.edges}
        start_nodes = set(self.nodes.keys()) - incoming_edges

        if not start_nodes:
            raise ValueError(
                "Graph must have at least one start node (node with no incoming edges)"
            )

        # Track visited nodes from all start nodes
        visited_nodes = set()

        # Check for cycles and track visited nodes from each start node
        for start_node in start_nodes:
            visited = set()
            stack = set()  # For cycle detection
            self._dfs_validate(start_node, visited, stack)
            visited_nodes.update(visited)

        # Check for orphaned nodes (nodes not reachable from any start node)
        orphaned_nodes = set(self.nodes.keys()) - visited_nodes
        if orphaned_nodes:
            raise ValueError(
                f"Found orphaned nodes not reachable from any start node: {orphaned_nodes}"
            )

        return True

    def _dfs_validate(self, node: str, visited: Set[str], stack: Set[str]) -> None:
        """Helper method for depth-first search validation.

        Args:
            node: Current node being visited
            visited: Set of nodes visited in this DFS traversal
            stack: Set of nodes in the current DFS stack (for cycle detection)

        Raises:
            ValueError: If a cycle is detected in the graph
        """
        if node in stack:
            raise ValueError(f"Cycle detected in graph involving node: {node}")

        if node in visited:
            return

        visited.add(node)
        stack.add(node)

        # Get all edges starting from current node
        outgoing_edges = {
            edge.end_node for edge in self.edges if edge.start_node == node
        }

        for next_node in outgoing_edges:
            self._dfs_validate(next_node, visited, stack)

        stack.remove(node)

    def compile(self, state: Union[BaseModel, NamedTuple, None] = None) -> Self:
        # Validate router nodes
        for node_name, node in self.nodes.items():
            if node.is_router and node.possible_routes:
                # Check if all possible routes exist as nodes
                invalid_routes = node.possible_routes - set(self.nodes.keys())
                if invalid_routes:
                    raise ValueError(
                        f"Router node '{node_name}' contains invalid routes: {invalid_routes}"
                    )

                # Check if all edges from this router point to declared possible routes
                router_edges = {
                    edge.end_node for edge in self.edges if edge.start_node == node_name
                }
                invalid_edges = router_edges - set(node.possible_routes)
                if invalid_edges:
                    raise ValueError(
                        f"Router node '{node_name}' has edges to undeclared routes: {invalid_edges}"
                    )

        for _, end in self.edges:
            self.tasks.append(self.nodes[end].action)

        self.is_compiled = True
        return self

    def visualize(self, output_file: str = "graph") -> None:
        """
        Visualize the graph using Graphviz.

        Args:
            output_file: Name of the output file (without extension)
        """
        from graphviz import Digraph

        dot = Digraph(comment="Graph Visualization")
        dot.attr(rankdir="LR")  # Left to right layout

        # Add nodes with styling
        for node_name, node in self.nodes.items():
            # Create label with metadata if it exists
            label = node_name
            if node.metadata or node.is_async or node.is_router:
                metadata_dict = node.metadata or {}
                if node.is_async:
                    metadata_dict = {**metadata_dict, "async": "true"}
                if node.is_router:
                    metadata_dict = {**metadata_dict, "router": "true"}
                    if node.possible_routes:
                        metadata_dict["routes"] = ",".join(node.possible_routes)
                metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata_dict.items())
                label = f"{node_name}\n{metadata_str}"

            dot.node(
                node_name,
                label,
                shape="box",
                style="rounded,filled",
                fillcolor="lightblue",
            )

        # Add edges
        for start, end in self.edges:
            dot.edge(start, end)

        # Render the graph
        dot.render(output_file, view=True, format="pdf", cleanup=True)


def _get_schema(state: Union[BaseModel, NamedTuple, None]) -> Dict[str, type]:
    if isinstance(state, BaseModel):
        pydantic_schema = {
            field_name: field_info.annotation
            for field_name, field_info in state.model_fields.items()
        }
        return pydantic_schema
    elif isinstance(state, tuple) and hasattr(
        state, "_fields"
    ):  # Check for named tuple
        return state.__annotations__
    return None
