from typing import Any, Callable, Dict, List, NamedTuple, Optional, Self, Set, Tuple

from tiny_graph.constants import END, START


class Edge(NamedTuple):
    start_node: str
    end_node: str


class Node(NamedTuple):
    name: str
    action: Callable[..., None]
    metadata: Optional[Dict[str, Any]] = None


class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Set[Edge] = set()
        self.is_compiled: bool = False

    @property
    def _all_nodes(self) -> List[str]:
        return list(self.nodes.keys())

    @property
    def _all_edges(self) -> Set[Tuple[str, str]]:
        return self.edges

    def add_node(
        self,
        name: str,
        action: Optional[Callable[..., None]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")
        if name == START or name == END:
            raise ValueError(f"Node {name} is reserved")
        if action is None:
            raise ValueError("Action cannot be None")
        if not callable(action):
            raise ValueError("Action must be a callable")

        self.nodes[name] = Node(name, action, metadata)
        return self

    def add_edge(self, start_node: str, end_node: str) -> Self:
        self.edges.add(Edge(start_node, end_node))
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
            if node.metadata:
                metadata_str = "\n".join(f"{k}: {v}" for k, v in node.metadata.items())
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
