import ast
import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Literal, NamedTuple, Optional, Self, Set, Tuple, Union

from pydantic import BaseModel

from primeGraph.constants import END, START

TUPLE_LENGTH = 2
MIN_VALID_NODES = 3  # START + END + at least one custom node


@dataclass(frozen=True)
class Edge:
  start_node: str
  end_node: str
  id: Optional[str]

  def __hash__(self) -> int:
    return hash((self.start_node, self.end_node))


class Node(NamedTuple):
  name: str
  action: Callable[..., None]
  metadata: Optional[Dict[str, Any]] = None
  is_async: bool = False
  is_router: bool = False
  possible_routes: Optional[Set[str]] = None
  interrupt: Union[Literal["before", "after"], None] = None
  emit_event: Optional[Callable] = None
  is_subgraph: bool = False
  subgraph: Optional["BaseGraph"] = None
  router_paths: Optional[Dict[str, List[str]]] = None


class BaseGraph:
  def __init__(self, state: Union[BaseModel, NamedTuple, None] = None):
    self.nodes: Dict[str, Node] = {}
    self.nodes[START] = Node(START, lambda: None, None, False, False, None)
    self.nodes[END] = Node(END, lambda: None, None, False, False, None)
    self.edges: Set[Edge] = set()
    self.is_compiled: bool = False
    self.tasks: List[Callable[..., None]] = []
    self.event_handlers: List[Callable] = []

    # Validate state type
    if state is not None and not isinstance(state, (BaseModel, NamedTuple)):
      raise TypeError("State must be either a Pydantic BaseModel, NamedTuple, or None")
    self.state: Union[BaseModel, NamedTuple, None] = state
    self._has_state = state is not None

    self.edge_counter: Dict[Tuple[str, str], int] = {}  # Track edge counts between node pairs

  @property
  def _all_nodes(self) -> List[str]:
    return list(self.nodes.keys())

  @property
  def _all_edges(self) -> Set[Edge]:
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
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
          return_values.add(node.value.value)
        # Handle string variable returns
        elif isinstance(node.value, ast.Name):
          # Check if the name matches our constants
          if node.value.id in ["START", "END"]:
            return_values.add("__" + node.value.id.lower() + "__")
        # Handle attribute access (for constants.START/END)
        elif isinstance(node.value, ast.Attribute) and node.value.attr in ["START", "END"]:
          return_values.add("__" + node.value.attr.lower() + "__")
    return return_values

  def _force_compile(self) -> None:
    if not self.is_compiled:
      print("Graph not compiled. Compiling now..")
      self.compile()

  def node(
    self,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    interrupt: Union[Literal["before", "after"], None] = None,
  ) -> Callable[..., None]:
    """Decorator to add a node to the graph

    Args:
        name: Optional name for the node. If None, uses the function name
        metadata: Optional metadata dictionary
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:  # noqa: ARG001, RUF100
      if self.is_compiled:
        raise ValueError("Cannot add nodes after compiling the graph")

      # Checking for reserved node names
      node_name = name if name is not None else func.__name__
      if node_name in [START, END]:
        raise ValueError(f"Node name '{node_name}' is reserved")

      # Checking for async calls
      is_async = inspect.iscoroutinefunction(func)

      # Create metadata as a dictionary attribute instead of using __metadata__
      func.metadata = {"interrupt": interrupt}  # type: ignore

      # Check if function accepts state parameter when graph has state
      if hasattr(self, "_has_state") and self._has_state:
        sig = inspect.signature(func)
        if "state" not in sig.parameters:
          raise ValueError(
            f"Node function '{func.__name__}' must accept 'state' parameter when graph has state. "
            f"Update your function definition to: def {func.__name__}(state) -> Dict"
          )

      # Create event emitter closure
      async def emit_event(event_type: str, data: Any = None) -> None:
        if hasattr(self, "event_handlers"):
          event = {
            "type": "node_event",
            "event_type": event_type,
            "node_id": node_name,
            "chain_id": getattr(self, "chain_id", None),
            "timestamp": datetime.now(),
            "data": data,
          }
          for handler in self.event_handlers:
            await handler(event)

      # Attach emit_event to the function
      func.emit_event = emit_event  # type: ignore

      self.nodes[node_name] = Node(
        node_name,
        func,
        metadata,
        is_async,
        False,
        None,
        interrupt,
        emit_event=emit_event,
      )
      return func

    return decorator  # type: ignore

  def subgraph(
    self,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    interrupt: Union[Literal["before", "after"], None] = None,
  ) -> Callable[[], "BaseGraph"]:
    """Decorator to add a subgraph as a node to the graph

    Args:
        name: Optional name for the subgraph node. If None, uses the function name
        metadata: Optional metadata dictionary
    """

    def decorator(func: Callable[[], "BaseGraph"]) -> Callable[[], "BaseGraph"]:
      if self.is_compiled:
        raise ValueError("Cannot add nodes after compiling the graph")

      # Use function name if name not provided
      node_name = name if name is not None else func.__name__

      # Check for reserved names
      if node_name in [START, END]:
        raise ValueError(f"Node name '{node_name}' is reserved")

      # Create a dummy action that will be replaced during execution
      def subgraph_action() -> None:
        pass

      # Add metadata about being a subgraph
      combined_metadata = {
        **(metadata or {}),
        "is_subgraph": True,
        "subgraph_type": "nested" if "_" in node_name else "parent",
      }

      # Execute the function to get the subgraph
      subgraph = func()

      # Create the node with the subgraph
      self.nodes[node_name] = Node(
        name=node_name,
        action=subgraph_action,
        metadata=combined_metadata,
        is_async=False,  # Will be determined during execution
        is_router=False,
        possible_routes=None,
        interrupt=interrupt,
        emit_event=None,
        is_subgraph=True,
        subgraph=subgraph,
      )
      return func

    return decorator  # type: ignore

  def add_edge(self, start_node: str, end_node: str, id: Optional[str] = None) -> Self:
    """Modified add_edge to handle subgraph connections"""
    if start_node not in self.nodes or end_node not in self.nodes:
      raise ValueError(
        f"Either start_node '{start_node}' or end_node '{end_node}' must be added to the graph before adding an edge"
      )

    # Check if either node is a subgraph
    start_is_subgraph = self.nodes[start_node].is_subgraph
    end_is_subgraph = self.nodes[end_node].is_subgraph

    # Handle subgraph connections
    if start_is_subgraph:
      subgraph = self.nodes[start_node].subgraph
      # Connect subgraph's END to the end_node
      if subgraph:
        self._merge_subgraph(subgraph, start_node, connect_end=end_node)
    elif end_is_subgraph:
      subgraph = self.nodes[end_node].subgraph
      # Connect start_node to subgraph's START
      if subgraph:
        self._merge_subgraph(subgraph, end_node, connect_start=start_node)
    else:
      # Normal edge connection
      if id is None:
        node_pair = (start_node, end_node)
        self.edge_counter[node_pair] = self.edge_counter.get(node_pair, 0) + 1
        id = f"{start_node}_to_{end_node}_{self.edge_counter[node_pair]}"

      self.edges.add(Edge(id=id, start_node=start_node, end_node=end_node))

    return self

  def _merge_subgraph(
    self,
    subgraph: "BaseGraph",
    subgraph_node: str,
    connect_start: Optional[str] = None,
    connect_end: Optional[str] = None,
  ) -> None:
    """Internal method to merge a subgraph into the main graph"""
    prefix = f"{subgraph_node}_"

    # Find first and last nodes in subgraph (excluding START and END)
    first_nodes = {edge.end_node for edge in subgraph.edges if edge.start_node == START}
    last_nodes = {edge.start_node for edge in subgraph.edges if edge.end_node == END}

    # Copy nodes from subgraph (excluding START and END)
    for node_name, node in subgraph.nodes.items():
      if node_name not in [START, END]:
        new_node_name = prefix + node_name
        # Create metadata that preserves subgraph hierarchy and pure name
        new_metadata = {
          **(node.metadata or {}),
          "parent_subgraph": subgraph_node,
          "is_subgraph": node.is_subgraph,
          "subgraph_type": "nested" if node.is_subgraph else "child",
          "subgraph_cluster": subgraph_node,
          "pure_name": node_name,  # Preserve the original node name
        }

        self.nodes[new_node_name] = Node(
          name=new_node_name,
          action=node.action,
          metadata=new_metadata,
          is_async=node.is_async,
          is_router=node.is_router,
          possible_routes=node.possible_routes,
          interrupt=node.interrupt,
          emit_event=node.emit_event,
          is_subgraph=node.is_subgraph,
          subgraph=node.subgraph,
        )

    # Copy and adjust edges
    for edge in subgraph.edges:
      if edge.start_node == START and connect_start:
        # Connect incoming edge to all first nodes of subgraph
        for first_node in first_nodes:
          self.add_edge(connect_start, prefix + first_node)
      elif edge.end_node == END and connect_end:
        # Connect all last nodes of subgraph to outgoing edge
        for last_node in last_nodes:
          self.add_edge(prefix + last_node, connect_end)
      elif edge.start_node != START and edge.end_node != END:
        self.add_edge(
          prefix + edge.start_node,
          prefix + edge.end_node,
          id=prefix + (edge.id or ""),
        )

  def validate(self) -> bool:
    """Validate that the graph starts with '__start__', ends with '__end__', and all nodes are on valid paths.

    Raises:
        ValueError: If start/end nodes are missing, orphaned nodes are found, dead ends exist, or if there are cycles
    """
    # Check for required start and end nodes
    if START not in self.nodes:
      raise ValueError("Graph must have a '__start__' node")
    if END not in self.nodes:
      raise ValueError("Graph must have an '__end__' node")

    # Track visited nodes starting from __start__
    visited: Set[str] = set()
    stack: Set[str] = set()  # For cycle detection
    self._dfs_validate(START, visited, stack)

    # Check if __end__ is reachable
    if END not in visited:
      raise ValueError("'__end__' node is not reachable from '__start__'")

    # Check for orphaned nodes (nodes not reachable from __start__)
    orphaned_nodes = set(self.nodes.keys()) - visited
    # Filter out subgraph nodes from orphaned nodes check
    non_subgraph_orphans = {node for node in orphaned_nodes if not self.nodes[node].is_subgraph}
    if non_subgraph_orphans:
      raise ValueError(f"Found orphaned nodes not reachable from '__start__': {non_subgraph_orphans}")

    # Check for graphs with only one node added or less
    if len(self.nodes) <= MIN_VALID_NODES:
      raise ValueError("Graph with only one node is not valid")

    # Check for dead ends (nodes with no outgoing edges, except END)
    nodes_with_outgoing_edges = {edge.start_node for edge in self.edges}
    dead_ends = set(self.nodes.keys()) - nodes_with_outgoing_edges - {END}
    # Filter out subgraph nodes from dead ends check
    non_subgraph_dead_ends = {node for node in dead_ends if not self.nodes[node].is_subgraph}
    if non_subgraph_dead_ends:
      raise ValueError(f"Found dead-end nodes with no outgoing edges: {non_subgraph_dead_ends}")

    # Validate router nodes
    for node_name, node in self.nodes.items():
      if node.is_router and node.possible_routes:
        # Check if all possible routes exist as nodes
        invalid_routes = node.possible_routes - set(self.nodes.keys())
        if invalid_routes:
          raise ValueError(f"Router node '{node_name}' contains invalid routes: {invalid_routes}")

        # Check if all edges from this router point to declared possible routes
        router_edges = {edge.end_node for edge in self.edges if edge.start_node == node_name}
        invalid_edges = router_edges - set(node.possible_routes)
        if invalid_edges:
          raise ValueError(f"Router node '{node_name}' has edges to undeclared routes: {invalid_edges}")

    return True

  def _dfs_validate(self, node: str, visited: Set[str], stack: Set[str]) -> None:
    """Helper method for depth-first search validation.

    Args:
        node: Current node being visited
        visited: Set of nodes visited in this DFS traversal
        stack: Set of nodes in the current DFS stack (for cycle detection)

    Raises:
        ValueError: If a cycle is detected in the graph that's not from a router return
    """
    if node in stack:
      # Check if the cycle is due to a router return
      is_router_return = any(
        node in (router_node.possible_routes or set()) for router_node in self.nodes.values() if router_node.is_router
      )
      if not is_router_return:
        raise ValueError(f"Cycle detected in graph involving node: {node}")
      return  # Allow the cycle if it's from a router return

    if node in visited:
      return

    visited.add(node)
    stack.add(node)

    # Get all edges starting from current node
    outgoing_edges = {edge.end_node for edge in self.edges if edge.start_node == node}

    for next_node in outgoing_edges:
      self._dfs_validate(next_node, visited, stack)

    stack.remove(node)

  def _find_execution_paths(self) -> List[Any]:
    """Identifies execution paths in the graph with parallel and nested parallel paths."""

    def get_next_nodes(node: str) -> List[str]:
      return [edge.end_node for edge in self.edges if edge.start_node == node]

    def get_prev_nodes(node: str) -> List[str]:
      return [edge.start_node for edge in self.edges if edge.end_node == node]

    def is_convergence_point(node: str) -> bool:
      return len(get_prev_nodes(node)) > 1

    def find_convergence_point(start_nodes: List[str]) -> str:
      """Find where multiple paths converge."""
      visited_by_path = {start: {start} for start in start_nodes}
      current_nodes = {start: [start] for start in start_nodes}
      max_iterations = len(self.nodes) * 2  # Safety limit
      iteration = 0

      while iteration < max_iterations:
        iteration += 1
        # For each path, get the next node
        for start in start_nodes:
          if current_nodes[start][-1] != END:
            current = current_nodes[start][-1]
            next_nodes = get_next_nodes(current)

            if not next_nodes:
              continue

            next_node = next_nodes[0]

            # Handle router nodes - don't follow cycles
            if self.nodes[current].is_router and next_node in visited_by_path[start]:
              continue

            visited_by_path[start].add(next_node)
            current_nodes[start].append(next_node)

        # Check if any node is visited by multiple paths
        all_visited: Set[str] = set()
        for nodes in visited_by_path.values():
          intersection = all_visited.intersection(nodes)
          if intersection:
            return next(iter(intersection))
          all_visited.update(nodes)

        # Check if all paths have reached END
        if all(nodes[-1] == END for nodes in current_nodes.values()):
          return END

      # If no convergence found within limit, return END
      return END

    def build_path_to_convergence(start: str, convergence: str, visited: Set[str], parent: str) -> List[Any]:  # noqa: PLR0912
      if start in visited:
        # If we've visited this node before, just add it without following its path
        return [(parent, start)]

      path = []
      current = start
      visited.add(current)

      while current != convergence:
        next_nodes = get_next_nodes(current)

        # Handle router nodes
        if current in self.nodes and self.nodes[current].is_router:
          path.append((parent, current))
          # Get the next node after the router
          if next_nodes:
            # Don't follow the path if it leads back to a visited node
            next_node = next_nodes[0]
            if next_node in visited:
              break
            current = next_node
            parent = current
            visited.add(current)
          continue

        if len(next_nodes) > 1:
          # Handle nested parallel paths
          nested_convergence = find_convergence_point(next_nodes)
          nested_paths = []
          for next_node in next_nodes:
            # Don't follow paths that lead back to visited nodes
            if next_node not in visited:
              nested_path = build_path_to_convergence(next_node, nested_convergence, visited.copy(), current)
              if nested_path:  # Only add non-empty paths
                if isinstance(nested_path, list) and len(nested_path) == 1:
                  nested_paths.append(nested_path[0])
                else:
                  nested_paths.append(nested_path)

          if nested_paths:  # Only extend if there are valid nested paths
            path.append([(parent, current), nested_paths])  # type: ignore
          current = nested_convergence
        elif not next_nodes:
          path.append((parent, current))
          visited.add(current)
          break
        elif is_convergence_point(current):
          path.append((parent, current))
          if next_nodes:
            current = next_nodes[0]
            parent = current
            visited.add(current)
          else:
            break
        else:
          path.append((parent, current))
          current = next_nodes[0]
          parent = current
          visited.add(current)

      return path

    def build_execution_plan(current: str, parent: str = START) -> List[Any]:
      if current == END:
        return []

      next_nodes = get_next_nodes(current)

      # Skip START node
      if current == START and len(next_nodes) == 1:
        return build_execution_plan(next_nodes[0], current)

      # Handle parallel paths
      if len(next_nodes) > 1:
        convergence = find_convergence_point(next_nodes)
        parallel_paths = []

        for next_node in next_nodes:
          path = build_path_to_convergence(next_node, convergence, set(), current)
          if isinstance(path, list) and len(path) == 1:
            path = path[0]
          parallel_paths.append(path)

        return [(parent, current), parallel_paths, *build_execution_plan(convergence, current)]

      # Handle sequential path
      return [(parent, current), *build_execution_plan(next_nodes[0], current)]

    # Build and clean the execution plan
    plan = build_execution_plan(START)

    def clean_plan(p: Any) -> Any:
      # Handle non-list items (like tuples)
      if not isinstance(p, list):
        return p

      # Handle empty lists
      if not p:
        return p

      # Handle single-item lists
      if len(p) == 1:
        return clean_plan(p[0])

      # Clean the list items
      return [
        clean_plan(item)
        for item in p
        if item is not None
        and (
          not isinstance(item, tuple)  # Handle non-tuple items
          or (len(item) == TUPLE_LENGTH and item[1] != START)  # Handle tuples
        )
      ]

    return clean_plan(plan)  # type: ignore

  def compile(self) -> Self:
    """Compiles the graph by validating and organizing execution paths."""
    self.validate()

    # Analyze router paths before creating execution paths
    self.router_paths = self._analyze_router_paths()

    self.detailed_execution_path = self._find_execution_paths()

    def extract_execution_plan(current_item: Any) -> Any:
      if isinstance(current_item, list):
        return [extract_execution_plan(item) for item in current_item]

      elif isinstance(current_item, str):
        return current_item
      else:
        return current_item[1]

    self.execution_path = [extract_execution_plan(item) for item in self.detailed_execution_path]

    self.is_compiled = True
    return self

  def visualize(self, transparent: bool = True) -> Any:  # noqa: PLR0912
    """Visualize the graph using Graphviz."""
    from graphviz import Digraph  # type: ignore

    def get_display_name(node_name: str, node: Node) -> Any:
      """Get clean display name from node metadata."""
      if node_name in [START, END]:
        return node_name
      if node.metadata is None:
        return node_name
      return node.metadata.get("pure_name", node_name)

    self._force_compile()

    dot = Digraph(comment="Graph Visualization", format="svg")
    # Balanced layout settings
    dot.attr(rankdir="TB")
    dot.attr(margin="0.3")  # Reduced margin further
    dot.attr(center="true")
    dot.attr(splines="ortho")
    dot.attr(ranksep="0.4")
    dot.attr(nodesep="0.3")  # Added to control horizontal spacing
    dot.attr("node", fontname="Helvetica", fontsize="10", margin="0.2,0.1")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # Add transparent background if requested
    if transparent:
      dot.attr(bgcolor="transparent")

    # Group nodes by subgraph, maintaining hierarchy
    subgraph_groups: Dict[str, Any] = {}
    repeat_groups: Dict[str, Any] = {}

    for node_name, node in self.nodes.items():
      # Skip subgraph container nodes
      if node.is_subgraph:
        continue

      # Handle repeated nodes
      if node.metadata and node.metadata.get("is_repeat"):
        group_id = node.metadata["repeat_group"]
        if group_id not in repeat_groups:
          repeat_groups[group_id] = {
            "nodes": set(),
            "original_node": node.metadata["original_node"],
            "parallel": node.metadata["parallel"],
            "parent_subgraph": node.metadata.get("parent_subgraph"),
          }
        repeat_groups[group_id]["nodes"].add(node_name)

      # Handle subgraph grouping (existing logic)
      if node.metadata and "parent_subgraph" in node.metadata:
        parent = node.metadata["parent_subgraph"]
        node_prefix = f"{parent}_"

        # Check if this node belongs to a nested subgraph
        if node_name.startswith(node_prefix + "inner_nested_"):
          nested_subgraph = f"{parent}_inner_nested"
          if parent not in subgraph_groups:
            subgraph_groups[parent] = {"nodes": set(), "nested": {}}
          if nested_subgraph not in subgraph_groups[parent]["nested"]:
            subgraph_groups[parent]["nested"][nested_subgraph] = set()
          subgraph_groups[parent]["nested"][nested_subgraph].add(node_name)
        else:
          if parent not in subgraph_groups:
            subgraph_groups[parent] = {"nodes": set(), "nested": {}}
          subgraph_groups[parent]["nodes"].add(node_name)

    def create_cluster(name: str, nodes: Set[str], nested: Dict[str, Any], parent_cluster: Any) -> None:
      with parent_cluster.subgraph(name=f"cluster_{name}") as cluster:
        cluster.attr(
          label=f"Subgraph: {name}",
          style="filled,rounded",
          fillcolor="#44444422",
          color="#AAAAAA",
          penwidth="1.0",
          fontname="Helvetica",
          fontcolor="#444444",
          margin="20",
          fontsize="10",  # Match node font size
          labelloc="t",  # Position at top
          labeljust="l",  # Left align
        )

        # Handle repeated nodes in this cluster
        repeat_clusters = {}
        for node_name in nodes:
          for group_id, group_info in repeat_groups.items():
            if (
              node_name in group_info["nodes"]
              and group_info["parent_subgraph"] == name
              and node_name == group_info["original_node"]
              and node_name not in repeat_clusters
            ):
              repeat_clusters[group_id] = group_info

        # Create repeat clusters first
        for group_id, group_info in repeat_clusters.items():
          with cluster.subgraph(name=f"cluster_repeat_{group_id}") as repeat_cluster:
            n_repeats = len(group_info["nodes"])
            execution_type = "Parallel" if group_info["parallel"] else "Sequential"
            repeat_cluster.attr(
              label=f"{n_repeats}x {execution_type}",
              style="dashed,rounded",
              color="#666666",
              fontcolor="#666666",
              fontname="Helvetica",
              fontsize="10",
              margin="8",
              labelloc="t",  # Place label at top
              labeljust="r",  # Align label to right
            )

            # Only add the original node to the visualization
            original_node_name = group_info["original_node"]
            if original_node_name in nodes:
              repeat_cluster.node(
                original_node_name,
                get_display_name(original_node_name, self.nodes[original_node_name]),
                style="rounded,filled",
                fillcolor="lightblue",
                shape="box",
              )

        # Add regular nodes that aren't part of repeat groups
        for node_name in nodes:
          if not any(
            (node_name in group["nodes"] and node_name != group["original_node"]) for group in repeat_groups.values()
          ):
            cluster.node(
              node_name,
              get_display_name(node_name, self.nodes[node_name]),
              style="rounded,filled",
              fillcolor="lightblue",
              shape="box",
            )

        # Create nested subgraph clusters
        for nested_name, nested_nodes in nested.items():
          create_cluster(nested_name, nested_nodes, {}, cluster)

    # Create all clusters with proper nesting
    for parent, group in subgraph_groups.items():
      create_cluster(parent, group["nodes"], group["nested"], dot)

    # Handle repeated nodes that aren't in subgraphs
    for group_id, group_info in repeat_groups.items():
      if not group_info["parent_subgraph"]:
        with dot.subgraph(name=f"cluster_repeat_{group_id}") as repeat_cluster:
          n_repeats = len(group_info["nodes"])
          execution_type = "Parallel" if group_info["parallel"] else "Sequential"
          repeat_cluster.attr(
            label=f"{n_repeats}x {execution_type}",
            style="dashed,rounded",
            color="#666666",
            fontcolor="#666666",
            fontname="Helvetica",
            fontsize="10",
            margin="8",
            labelloc="t",  # Place label at top
            labeljust="r",  # Align label to right
          )

          # Only add the original node
          original_node_name = group_info["original_node"]
          repeat_cluster.node(
            original_node_name,
            get_display_name(original_node_name, self.nodes[original_node_name]),
            style="rounded,filled",
            fillcolor="lightblue",
            shape="box",
          )

    def create_interrupt_node(dot: Any, node_name: str, interrupt: str) -> None:
      """Create a small red circle indicator for interrupt."""
      display_name = get_display_name(node_name, self.nodes[node_name])
      # Use a medium dot character and add italic to the interrupt label
      interrupt_label = f'<FONT COLOR="#AA0000">●</FONT> <I>Int. {interrupt}</I>'

      if interrupt == "before":
        node_label = f"{interrupt_label} | {display_name}"
      else:  # after
        node_label = f"{display_name} | {interrupt_label}"

      dot.node(
        node_name,
        f"<{node_label}>",  # Wrap in <> to enable HTML-like formatting
        style="rounded,filled",
        fillcolor="lightblue",
        shape="record",
        fontname="Helvetica",
        color="#666666",
      )

    # Update the node creation section in visualize method:
    for node_name, node in self.nodes.items():
      if (
        not node.is_subgraph
        and not (node.metadata and "parent_subgraph" in node.metadata)
        and not any(
          (node_name in group["nodes"] and node_name != group["original_node"]) for group in repeat_groups.values()
        )
      ):
        if node_name in [START, END]:
          dot.node(node_name, node_name, shape="ellipse", fillcolor="#F4E8E8", style="filled")
        elif node.interrupt:
          create_interrupt_node(dot, node_name, node.interrupt)
        else:
          dot.node(
            node_name,
            get_display_name(node_name, node),
            style="rounded,filled",
            fillcolor="lightblue",
            shape="box",
          )

    # Add edges (only for visible nodes)
    added_visual_edges = set()  # Track edges we've added for visualization
    for edge in self.edges:
      if not self.nodes[edge.start_node].is_subgraph and not self.nodes[edge.end_node].is_subgraph:
        # Check if this is a repeated node edge
        start_metadata = self.nodes[edge.start_node].metadata or {}
        end_metadata = self.nodes[edge.end_node].metadata or {}

        # Handle repeated node connections
        if start_metadata.get("is_repeat"):
          original_node = start_metadata["original_node"]

          # If this is the last repeated node connecting to an end node
          if not end_metadata.get("is_repeat"):
            # Add visual edge from original node to end node if not already added
            visual_edge = (original_node, edge.end_node)
            if visual_edge not in added_visual_edges:
              dot.edge(
                original_node,
                edge.end_node,
                style="dashed",
                color="#666666",
              )
              added_visual_edges.add(visual_edge)
          continue  # Skip adding the actual repeated node edge

        # Add normal edges that aren't from/to repeated nodes (except original)
        if not any(
          (node in group["nodes"] and node != group["original_node"])
          for group in repeat_groups.values()
          for node in (edge.start_node, edge.end_node)
        ):
          # Check if the edge starts from a router node
          is_router_edge = self.nodes[edge.start_node].is_router
          edge_style = "dashed" if is_router_edge else "solid"
          edge_color = "#666666" if is_router_edge else "black"

          dot.edge(
            edge.start_node,
            edge.end_node,
            style=edge_style,
            color=edge_color,
          )

    # Update node visualization for router nodes
    for node_name, node in self.nodes.items():
      if node.is_router:
        node_attrs = {
          "style": "rounded, filled",
          "fillcolor": "#FFE4B5",  # Light orange for router nodes
          "shape": "diamond",
          "margin": "0.2",  # Added margin inside node
          "penwidth": "1.0",
          "regular": "true",  # Makes the diamond more regular/symmetric
          "width": "1.5",  # Increased width
          "height": "1.5",  # Increased height
          "fixedsize": "false",  # Allow node to resize based on label
        }

        # Format the label to ensure it fits
        display_name = get_display_name(node_name, node)
        formatted_label = display_name.replace("_", "\n")  # Break long names with newlines

        dot.node(node_name, formatted_label, **node_attrs)

        # Add edge labels for router paths if available
        if node.router_paths:
          for first_node, _ in node.router_paths.items():
            edge = next(e for e in self.edges if e.start_node == node_name and e.end_node == first_node)
            if edge:
              dot.edge(
                edge.start_node,
                edge.end_node,
                label=f"route: {first_node}",
                style="dashed",
                color="#666666",
              )

    return dot

  def find_edges(
    self,
    start_node: Optional[str] = None,
    end_node: Optional[str] = None,
    edge_id: Optional[str] = None,
  ) -> Set[Edge]:
    """Find edges matching the given criteria"""
    result = self.edges

    if edge_id:
      result = {edge for edge in result if edge.id == edge_id}
    if start_node:
      result = {edge for edge in result if edge.start_node == start_node}
    if end_node:
      result = {edge for edge in result if edge.end_node == end_node}

    return result

  def add_repeating_edge(
    self,
    start_node: str,
    repeat_node: str,
    end_node: str,
    repeat: int = 1,
    parallel: bool = False,
  ) -> Self:
    """Add a repeating edge that creates multiple instances of the same node.

    Args:
        start_node: Starting node name
        repeat_node: Node to be repeated
        end_node: Ending node name
        repeat: Number of times to repeat the node
        parallel: Whether to run repetitions in parallel
    """
    if repeat < 1:
      raise ValueError("Repeat count must be at least 1")

    if start_node not in self.nodes or end_node not in self.nodes or repeat_node not in self.nodes:
      raise ValueError("All nodes must exist in the graph")

    if start_node == START:
      raise ValueError("Repeating nodes cannot have START as a start node. It needs to be connected to another node.")
    node_obj = self.nodes[start_node]
    if not node_obj:
      raise ValueError(f"Node {start_node} does not exist in the graph")

    if node_obj.metadata and node_obj.metadata.get("is_repeat"):
      raise ValueError("Repeating nodes cannot have another repeating node as a start node.")

    # Get the original node to be repeated
    original_node = self.nodes[repeat_node]

    # Create a unique short ID using the edge count
    short_id = f"{len(self.edges):03x}"  # Using hex to keep it shorter

    # Create a wrapper function that maintains node identity
    def create_node_action(
      node_name: str, original_action: Callable
    ) -> Callable[..., Union[Any, Coroutine[Any, Any, Any]]]:
      if inspect.iscoroutinefunction(original_action):

        async def wrapped_action(*args: Any, **kwargs: Any) -> Any:
          return await original_action(*args, **kwargs)
      else:

        def wrapped_action(*args: Any, **kwargs: Any) -> Any:  # type: ignore
          return original_action(*args, **kwargs)

      # Set the name and node_name attributes
      wrapped_action.__name__ = node_name
      wrapped_action.__node_name__ = node_name  # type: ignore

      # Copy all other attributes from the original action
      for attr in dir(original_action):
        if not attr.startswith("__"):
          setattr(wrapped_action, attr, getattr(original_action, attr))

      return wrapped_action

    # Create a new version of the original node with updated metadata
    self.nodes[repeat_node] = Node(
      name=repeat_node,
      action=create_node_action(repeat_node, original_node.action),  # type: ignore
      metadata={
        **(original_node.metadata or {}),
        "is_repeat": True,
        "repeat_group": short_id,
        "repeat_index": 1,
        "original_node": repeat_node,
        "parallel": parallel,
      },
      is_async=original_node.is_async,
      is_router=original_node.is_router,
      possible_routes=original_node.possible_routes,
      interrupt=original_node.interrupt,
      emit_event=original_node.emit_event,
      is_subgraph=original_node.is_subgraph,
      subgraph=original_node.subgraph,
    )

    repeated_nodes = [repeat_node]

    # Create n-1 additional repeated nodes
    for i in range(repeat - 1):
      repeat_node_name = f"{repeat_node}_{i+2}_{short_id}"  # e.g., node_2_001, node_3_001

      self.nodes[repeat_node_name] = Node(
        name=repeat_node_name,
        action=create_node_action(repeat_node_name, original_node.action),  # type: ignore
        metadata={
          **(original_node.metadata or {}),
          "is_repeat": True,
          "repeat_group": short_id,
          "repeat_index": i + 2,
          "original_node": repeat_node,
          "parallel": parallel,
        },
        is_async=original_node.is_async,
        is_router=original_node.is_router,
        possible_routes=original_node.possible_routes,
        interrupt=original_node.interrupt,
        emit_event=original_node.emit_event,
        is_subgraph=original_node.is_subgraph,
        subgraph=original_node.subgraph,
      )
      repeated_nodes.append(repeat_node_name)

    # Connect the nodes based on parallel/sequential execution
    if parallel:
      # Connect start node to all repeated nodes
      for node_name in repeated_nodes:
        self.add_edge(start_node, node_name)

      # Connect all repeated nodes to end node
      for node_name in repeated_nodes:
        self.add_edge(node_name, end_node)
    else:
      # Connect nodes sequentially
      self.add_edge(start_node, repeated_nodes[0])
      for i in range(len(repeated_nodes) - 1):
        self.add_edge(repeated_nodes[i], repeated_nodes[i + 1])
      self.add_edge(repeated_nodes[-1], end_node)

    return self

  def add_router_edge(self, start_node: str, router_node: str) -> Self:
    """Add a router edge that can direct flow to different paths based on router node return value.

    Args:
        start_node: Starting node name
        router_node: Node that will determine the routing
    """
    if not all(node in self.nodes for node in [start_node, router_node]):
      raise ValueError("All nodes must exist in the graph")

    # Update router node metadata
    router_metadata = self.nodes[router_node].metadata or {}
    router_metadata.update({"is_router": True})

    # Get all possible return values from the router function
    return_values = self._get_return_values(self.nodes[router_node].action)

    if not return_values:
      raise ValueError(f"Router node '{router_node}' must return string literals indicating next nodes")

    # Create new Node instance with updated metadata and possible routes
    self.nodes[router_node] = self.nodes[router_node]._replace(
      is_router=True, metadata=router_metadata, possible_routes=return_values
    )

    # Add edge from start to router
    self.add_edge(start_node, router_node)

    # Automatically add edges to all possible return nodes
    for return_node in return_values:
      if return_node not in self.nodes:
        raise ValueError(f"Return node '{return_node}' does not exist in the graph")
      self.add_edge(router_node, return_node)

    return self

  def _analyze_router_paths(self) -> Dict[str, Dict[str, List[str]]]:
    """Identifies execution paths in the graph with parallel and nested parallel paths."""
    router_paths: Dict[str, Dict[str, List[str]]] = {}

    def get_next_nodes(node: str) -> List[str]:
      return [edge.end_node for edge in self.edges if edge.start_node == node]

    def get_prev_nodes(node: str) -> List[str]:
      return [edge.start_node for edge in self.edges if edge.end_node == node]

    def is_convergence_point(node: str) -> bool:
      return len(get_prev_nodes(node)) > 1

    def follow_path(start_node: str, visited: Optional[Set[str]] = None) -> List[str]:
      if visited is None:
        visited = set()

      if start_node in visited:
        return []

      current_path = [start_node]
      current = start_node
      visited.add(current)

      while True:
        next_nodes = get_next_nodes(current)
        if not next_nodes or current == END:
          break

        next_node = next_nodes[0]
        # Check if next node is a router
        if next_node in self.nodes and self.nodes[next_node].is_router:
          # Add the router node but don't follow its paths
          current_path.append(next_node)
          break

        if is_convergence_point(next_node):
          current_path.append(next_node)
          break

        current_path.append(next_node)
        current = next_node
        visited.add(current)

      return current_path

    # First pass to collect all router paths
    for node_name, node in self.nodes.items():
      if node.is_router and node.possible_routes:
        paths: Dict[str, List[str]] = {}
        for route in node.possible_routes:
          current_path = follow_path(route)
          paths[route] = current_path

        router_paths[node_name] = paths

    # Second pass to handle nested routers
    for _, paths in router_paths.items():
      for route, path in paths.items():
        complete_path: List[str] = []
        for node_name in path:
          complete_path.append(node_name)
          if node_name in router_paths:
            # If this node is a router, get its default path
            next_route = next(iter(router_paths[node_name].keys()))
            complete_path.extend(router_paths[node_name][next_route][1:])
        paths[route] = complete_path

    return router_paths
