from typing import Dict, Set, Tuple


class Graph:
    def __init__(self):
        self.nodes: Dict[str, str] = {}
        self.edges: Set[Tuple[str, str]] = set()

    def add_node(self, node: str):
        self.nodes[node] = node

    def add_edge(self, edge: Tuple[str, str]):
        self.edges.add(edge)
