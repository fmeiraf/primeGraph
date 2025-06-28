"""
Module for LLM tool nodes that support function calling patterns.

This module provides a specialized node type for LLM interactions with tools/function calling,
operating in a loop pattern. It integrates with the primeGraph system while providing
a separate execution path specifically designed for LLM tool interactions.
"""

import asyncio
import inspect
import json
import logging
import time
import traceback
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, Field

from primeGraph.buffer.factory import History, LastValue
from primeGraph.checkpoint.base import CheckpointData, StorageBackend
from primeGraph.constants import END, START
from primeGraph.graph.base import Node
from primeGraph.graph.engine import Engine, ExecutionFrame
from primeGraph.graph.executable import Graph
from primeGraph.graph.llm_clients import LLMMessage, StreamingConfig, StreamingEventType
from primeGraph.graph.tool_validation import validate_tool_args
from primeGraph.models.state import GraphState
from primeGraph.types import ChainStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    """Types of tools that can be used with LLMs"""

    FUNCTION = "function"
    ACTION = "action"  # For tools that perform actions but may not return values
    RETRIEVAL = "retrieval"  # For retrieval/search tools


class ToolCallStatus(str, Enum):
    """Status of a tool call execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by LLMs"""

    name: str
    description: str
    type: ToolType = ToolType.FUNCTION
    parameters: Dict[str, Any] = {}
    required_params: List[str] = []
    hidden_params: List[str] = []  # List of parameter names to hide from schema
    func: Optional[Callable] = None
    pause_before_execution: bool = False  # Flag to pause execution before this tool runs
    pause_after_execution: bool = False  # Flag to pause execution after this tool runs
    abort_after_execution: bool = False  # Flag to abort the loop immediately after this tool runs

    model_config = {"arbitrary_types_allowed": True}


class ToolUseRecord(BaseModel):
    """Record of a tool use, stored in state"""

    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    status: ToolCallStatus
    timestamp: float
    duration_ms: float = 0.0
    error: Optional[str] = None


class ToolCallLog(BaseModel):
    """Log entry for a tool call within a loop"""

    id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    timestamp: float
    duration_ms: float = 0.0
    error: Optional[str] = None


class ToolLoopOptions(BaseModel):
    """Options for configuring a tool loop execution"""

    max_iterations: int = 10
    timeout_seconds: Optional[float] = None
    max_tokens: int = 4096
    stop_on_first_error: bool = False
    trace_enabled: bool = False
    model: Optional[str] = None
    api_kwargs: Dict[str, Any] = Field(default_factory=dict)
    streaming_config: Optional[StreamingConfig] = None
    # Streaming shortcut flags (will create a StreamingConfig if streaming_config is None)
    stream: bool = False
    stream_events: Optional[Set[StreamingEventType]] = None
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_channel: Optional[str] = None
    stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    model_config = {"arbitrary_types_allowed": True}


class ToolState(GraphState):
    """Base state for tool loops, storing messages, tool calls, and results"""

    messages: History[LLMMessage] = Field(default_factory=lambda: [])  # type: ignore
    tool_calls: History[ToolCallLog] = Field(default_factory=lambda: [])  # type: ignore
    current_iteration: LastValue[int] = 0  # type: ignore
    max_iterations: LastValue[int] = 10  # type: ignore
    is_complete: LastValue[bool] = False  # type: ignore
    final_output: LastValue[Optional[Any]] = None  # type: ignore
    error: LastValue[Optional[str]] = None  # type: ignore
    current_trace: LastValue[Optional[Dict[str, Any]]] = None  # type: ignore
    raw_response_history: History[Any] = Field(default_factory=lambda: [], exclude=False)  # type: ignore
    is_paused: LastValue[bool] = False  # type: ignore
    paused_tool_id: LastValue[Optional[str]] = None  # type: ignore
    paused_tool_name: LastValue[Optional[str]] = None  # type: ignore
    paused_tool_arguments: LastValue[Optional[Dict[str, Any]]] = None  # type: ignore
    paused_after_execution: LastValue[bool] = False  # type: ignore
    paused_tool_result: LastValue[Optional[ToolCallLog]] = None  # type: ignore
    # Streaming-related fields
    streaming_enabled: LastValue[bool] = False  # type: ignore
    streaming_channel: LastValue[Optional[str]] = None  # type: ignore
    last_stream_timestamp: LastValue[Optional[float]] = None  # type: ignore


def tool(
    description: str,
    tool_type: ToolType = ToolType.FUNCTION,
    pause_before_execution: bool = False,
    pause_after_execution: bool = False,
    abort_after_execution: bool = False,
    hidden_params: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator for tool functions.

    This decorator creates a ToolDefinition for a function, which can
    then be used by ToolNode to call the function on demand based on LLM
    tool use requests.

    Args:
        description: Description of what the tool does.
        tool_type: Type of tool (function, action, retrieval).
        pause_before_execution: Whether to pause before executing the tool.
        pause_after_execution: Whether to pause after executing the tool.
        abort_after_execution: Whether to abort the loop immediately after executing the tool.
        hidden_params: List of parameter names to hide from the schema.

    Returns:
        Decorated function with tool metadata.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        parameters = {}
        required_params = []

        for name, param in sig.parameters.items():
            if name == "state":
                # Skip 'state' parameter as it's injected by the wrapper
                continue

            param_schema: Dict[str, Any] = {"type": "string"}  # Default to string

            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if isinstance(param.annotation, str):
                    param_schema = {"type": "string"}
                elif isinstance(param.annotation, int):
                    param_schema = {"type": "integer"}
                elif isinstance(param.annotation, float):
                    param_schema = {"type": "number"}
                elif isinstance(param.annotation, bool):
                    param_schema = {"type": "boolean"}
                elif isinstance(param.annotation, list) and isinstance(param.annotation.__args__[0], str):
                    param_schema = {"type": "array", "items": {"type": "string"}}
                elif isinstance(param.annotation, dict):
                    param_schema = {"type": "object"}
                # Add more type handling as needed

            # Handle default values
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
            else:
                required_params.append(name)

            parameters[name] = param_schema

        # Create the tool definition object and attach it to the function
        func._tool_definition = ToolDefinition(
            name=func.__name__,
            description=description,
            type=tool_type,
            parameters=parameters,
            required_params=required_params,
            hidden_params=hidden_params or [],  # Use empty list if None
            func=func,
            pause_before_execution=pause_before_execution,
            pause_after_execution=pause_after_execution,
            abort_after_execution=abort_after_execution,
        )

        # Create a wrapper that injects the state parameter if the function supports it
        @wraps(func)
        async def wrapper(*args, state=None, **kwargs: Any) -> Any:
            """Wrapper that injects state if the function accepts it"""
            sig = inspect.signature(func)

            if "state" in sig.parameters and state is not None:
                # Function accepts state parameter, inject it
                return await func(*args, state=state, **kwargs)
            else:
                # Function doesn't use state, call normally
                return await func(*args, **kwargs)

        # Add the tool definition to the wrapper
        wrapper._tool_definition = func._tool_definition

        return wrapper

    return decorator


class ToolNode(Node):
    """
    A specialized node for LLM tool interaction loops.

    This node type runs an LLM with a set of tools in a loop pattern,
    capturing all interactions and maintaining state throughout the loop.
    """

    def __new__(
        cls,
        name: str,
        tools: List[Callable],
        llm_client: Any,  # LLMClientBase instance
        options: Optional[ToolLoopOptions] = None,
        state_class: Type[GraphState] = ToolState,
        on_tool_use: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Create a new ToolNode instance with the appropriate NamedTuple structure."""

        # Create a dummy action function that will be properly handled in the engine
        async def tool_action(state: GraphState) -> Dict[str, Any]:
            # This is a placeholder - execution is handled by ToolEngine
            return {}

        # Define metadata for the tool node
        metadata = {
            "tool_count": len(tools),
            "options": options.model_dump() if options else {},
        }

        # Create the Node instance, marking it as async since tool execution is async
        instance = super().__new__(
            cls,
            name=name,
            action=tool_action,
            metadata=metadata,
            is_async=True,
            is_router=False,
            possible_routes=None,
            interrupt=None,
            emit_event=None,
            is_subgraph=False,
            subgraph=None,
            router_paths=None,
        )

        # Add tool node specific attributes
        instance.tools = tools
        instance.llm_client = llm_client
        instance.options = options or ToolLoopOptions()
        instance.state_class = state_class
        instance.on_tool_use = on_tool_use
        instance.on_message = on_message
        instance.is_tool_node = True

        # Validate tools
        for i, tool_func in enumerate(instance.tools):
            if not hasattr(tool_func, "_tool_definition"):
                raise ValueError(f"Tool at index {i} ({tool_func.__name__}) is not decorated with @tool")

        return instance

    # Original validate_tools method moved to __new__

    def get_tool_schemas(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get schema definitions for all tools, formatted for the specified provider.

        Args:
            provider: LLM provider name ('openai', 'anthropic', etc.)

        Returns:
            List of tool schema definitions
        """
        schemas = []
        for tool_func in self.tools:
            tool_def = tool_func._tool_definition

            # Create parameter schema for JSON schema format
            json_schema = {"type": "object", "properties": {}, "required": []}

            # Add each parameter to the JSON schema, excluding hidden ones
            for param_name, param_info in tool_def.parameters.items():
                if param_name not in tool_def.hidden_params:
                    json_schema["properties"][param_name] = param_info
                    # Only add to required if it's required and not hidden
                    if param_name in tool_def.required_params:
                        json_schema["required"].append(param_name)

            # Format for provider
            if provider and provider.lower() == "anthropic":
                # Anthropic format directly matches their API requirements
                schema = {"name": tool_def.name, "description": tool_def.description, "input_schema": json_schema}
            elif provider and provider.lower() == "google":
                schema = {"name": tool_def.name, "description": tool_def.description, "parameters": json_schema}
            else:
                # Default to OpenAI format
                schema = {
                    "type": "function",
                    "function": {"name": tool_def.name, "description": tool_def.description, "parameters": json_schema},
                }

            schemas.append(schema)

        return schemas

    def find_tool_by_name(self, name: str) -> Optional[Callable]:
        """Find a tool by name from this node's tool list"""
        for tool_func in self.tools:
            if tool_func._tool_definition.name == name:
                return tool_func
        return None

    def convert_tool_result_to_message(self, tool_results: ToolCallLog) -> Dict[str, Any]:
        """
        Get the response from the tool call.
        """
        if self.llm_client.provider == "anthropic":
            return LLMMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_results.id,
                        "content": str(tool_results.result),
                    }
                ],
            )
        else:
            # default to openai format
            return LLMMessage(
                role="tool",
                content=str(tool_results.result),
                tool_call_id=tool_results.id,
            )

    async def execute_tool(
        self, tool_func: Callable, arguments: Dict[str, Any], tool_id: str, state: Optional[GraphState] = None
    ) -> ToolCallLog:
        """Execute a tool function with the given arguments."""
        start_time = time.time()
        tool_name = getattr(tool_func, "__name__", "unknown")

        try:
            # Validate arguments before execution
            validated_args = validate_tool_args(tool_func, arguments)

            # Check if the tool can accept a state parameter
            sig = inspect.signature(tool_func)
            if "state" in sig.parameters and state is not None:
                # Pass the current state along with other arguments
                result = await tool_func(state=state, **validated_args)
            else:
                # Call without state
                result = await tool_func(**validated_args)

            success = True
            error = None
        except ValueError as e:
            # Handle validation errors
            result = None
            success = False
            error = f"Validation error for tool {tool_name}: {str(e)}"
            traceback.print_exc()
        except Exception as e:
            # Handle other execution errors
            result = None
            success = False
            error = f"Error executing tool {tool_name}: {str(e)}"
            traceback.print_exc()

        duration_ms = (time.time() - start_time) * 1000

        tool_result = ToolCallLog(
            id=tool_id,
            tool_name=tool_name,
            arguments=arguments,  # Keep original arguments for logging
            result=result,
            success=success,
            timestamp=start_time,
            duration_ms=duration_ms,
            error=error,
        )

        tool_message = self.convert_tool_result_to_message(tool_result)

        return tool_result, tool_message


class ToolGraph(Graph):
    """
    A graph specialized for executing LLM-based tool workflows.

    This graph extension provides additional methods for creating and connecting
    tool nodes to enable tool-based workflows with LLMs.

    ToolGraph follows the same pattern as the base Graph class:
    - It has a ToolEngine as a state variable (self.execution_engine)
    - It provides execute() and resume() methods that delegate to the engine
    - It handles state management and checkpointing

    This allows for a consistent interface across all graph types.
    """

    def __init__(
        self,
        name: str,
        state: Optional[GraphState] = None,
        max_iterations: int = 10,
        checkpoint_storage: Optional[StorageBackend] = None,
        execution_timeout: int = 60 * 5,
        max_node_iterations: int = 100,
        verbose: bool = False,
        output_schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize a tool-specific graph.

        Args:
            name: Name of the graph
            state: Optional state instance for the graph
            max_iterations: Maximum iterations for tool loops
            checkpoint_storage: Optional storage backend for checkpoints
            execution_timeout: Timeout for execution in seconds
            max_node_iterations: Maximum iterations per node
            verbose: Whether to enable verbose logging
            output_schema: Optional Pydantic BaseModel class defining expected output format
            system_prompt: Optional custom system prompt for the LLM
        """
        # Initialize state from the state class

        # Set graph name and state class as properties
        self.name = name
        self.state = state if state is not None else ToolState()
        self.state_class = state.__class__ if state is not None else ToolState

        # Store schema and prompt settings
        self.output_schema = output_schema
        self.system_prompt = system_prompt or ""

        # Build the complete system prompt
        self._build_system_prompt()

        # Call parent constructor with all required parameters
        super().__init__(
            state=self.state,
            checkpoint_storage=checkpoint_storage,
            execution_timeout=execution_timeout,
            max_node_iterations=max_node_iterations,
            verbose=verbose,
        )

        # Replace the default Engine with a ToolEngine
        self.execution_engine = ToolEngine(
            graph=self, timeout=execution_timeout, max_node_iterations=max_node_iterations
        )

        self.max_iterations = max_iterations
        self.final_output = None

        # START and END nodes are added in the BaseGraph constructor

    def _build_system_prompt(self) -> None:
        """Build the complete system prompt with schema instructions if needed."""
        # Start with user's custom prompt or default
        if not self.system_prompt.strip():
            self.system_prompt = "You are a helpful AI assistant that can use tools to accomplish tasks."

        # Add schema instructions if output_schema is provided
        if self.output_schema:
            try:
                # Generate JSON schema from Pydantic model
                json_schema = self.output_schema.model_json_schema()

                schema_instructions = f"""

                Always respond with a JSON object that's compatible with this schema:
                {json.dumps(json_schema, indent=2)}
                Don't include any text or Markdown fencing before or after.

                ALWAYS use the tool shared with you that ends the conversation to send this message."""

                self.system_prompt += schema_instructions
            except Exception as e:
                print(f"Warning: Could not generate JSON schema from output_schema: {e}")

    def _create_end_conversation_tool(self) -> Callable:
        """Create the automatic end_conversation tool."""

        @tool(
            description="End the conversation with the final output. "
            "Use this when you have completed the task and want to provide the final result.",
            tool_type=ToolType.ACTION,
            abort_after_execution=True,
        )
        async def end_conversation(final_output: str, state=None) -> str:
            """
            End the conversation with the final output.

            Args:
                final_output: The final result/answer to provide
                state: Optional state parameter (injected automatically)

            Returns:
                The validated final output
            """
            # If we have an output schema, validate the final_output against it
            if self.output_schema:
                try:
                    # Try to parse the final_output as JSON and validate against schema
                    import json

                    # Parse the JSON string
                    if isinstance(final_output, str):
                        try:
                            parsed_data = json.loads(final_output)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                "Final output must be valid JSON when output_schema " f"is specified. Error: {e}"
                            )
                    else:
                        parsed_data = final_output

                    # Validate against the schema
                    validated_result = self.output_schema(**parsed_data)

                    # Convert back to JSON string for consistency
                    return validated_result.model_dump_json()

                except Exception as e:
                    raise ValueError(f"Final output validation failed: {e}")
            else:
                # No schema validation needed, return as-is
                return final_output

        return end_conversation

    def add_tool_node(
        self,
        name: str,
        tools: List[Callable],
        llm_client: Any,
        options: Optional[ToolLoopOptions] = None,
        on_tool_use: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_connect: bool = True,
    ) -> ToolNode:
        """
        Add a tool node to the graph.

        Args:
            name: Name of the node
            tools: List of tool functions
            llm_client: LLM client instance for provider API calls
            options: Tool loop options
            on_tool_use: Optional callback for tool use
            on_message: Optional callback for non-tool LLM messages
            auto_connect: Whether to automatically connect this node to START/END if it's the only tool node

        Returns:
            The created ToolNode
        """
        if options is None:
            options = ToolLoopOptions(max_iterations=self.max_iterations)

        # Create a copy of tools and add the automatic end_conversation tool
        enhanced_tools = tools.copy()
        enhanced_tools.append(self._create_end_conversation_tool())

        node = ToolNode(
            name=name,
            tools=enhanced_tools,
            llm_client=llm_client,
            options=options,
            state_class=self.state_class,
            on_tool_use=on_tool_use,
            on_message=on_message,
        )

        # Store reference to system prompt and output schema on the node
        node.system_prompt = self.system_prompt
        node.output_schema = self.output_schema

        # Add the node directly to the nodes dictionary
        self.nodes[name] = node

        # Auto-connect if requested and this is the only tool node
        if auto_connect:
            self._auto_connect_if_single_node(node)

        return node

    def add_single_tool_workflow(
        self,
        name: str,
        tools: List[Callable],
        llm_client: Any,
        options: Optional[ToolLoopOptions] = None,
        on_tool_use: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ToolNode:
        """
        Convenience method to add a single tool node workflow with automatic START/END connections.

        This method automatically connects START -> tool_node -> END, making it perfect for
        simple single-node tool workflows.

        Args:
            name: Name of the node
            tools: List of tool functions
            llm_client: LLM client instance for provider API calls
            options: Tool loop options
            on_tool_use: Optional callback for tool use
            on_message: Optional callback for non-tool LLM messages

        Returns:
            The created ToolNode
        """
        # Clear any existing tool nodes to ensure this is truly a single-node workflow
        tool_nodes = self._get_tool_nodes()
        if tool_nodes:
            raise ValueError(
                f"Cannot create single tool workflow: {len(tool_nodes)} tool nodes already exist. "
                "Use add_tool_node() instead."
            )

        # Add the node without auto-connect first
        node = self.add_tool_node(
            name=name,
            tools=tools,
            llm_client=llm_client,
            options=options,
            on_tool_use=on_tool_use,
            on_message=on_message,
            auto_connect=False,
        )

        # Force connection to START and END
        self.add_edge(START, node.name)
        self.add_edge(node.name, END)

        return node

    def _get_tool_nodes(self) -> List[ToolNode]:
        """Get all tool nodes in the graph."""
        tool_nodes = []
        for node in self.nodes.values():
            if hasattr(node, "is_tool_node") and node.is_tool_node:
                tool_nodes.append(node)
        return tool_nodes

    def _auto_connect_if_single_node(self, node: ToolNode) -> None:
        """
        Automatically connect a node to START and END if it's the only tool node.

        Args:
            node: The tool node to potentially connect
        """
        tool_nodes = self._get_tool_nodes()

        # Only auto-connect if this is the only tool node
        if len(tool_nodes) == 1:
            # Check if START and END connections already exist
            start_children = self.edges_map.get(START, [])
            node_children = self.edges_map.get(node.name, [])

            # Connect START -> node if not already connected
            if node.name not in start_children:
                self.add_edge(START, node.name)

            # Connect node -> END if not already connected
            if END not in node_children:
                self.add_edge(node.name, END)

    def ensure_connected_workflow(self) -> None:
        """
        Ensure that the tool graph has a connected workflow from START to END.

        This method will automatically connect any unconnected tool nodes:
        - If there's only one tool node, it connects START -> node -> END
        - If there are multiple tool nodes, it connects them in a linear chain
        - If nodes are already connected, it leaves them alone

        Raises:
            ValueError: If the graph has no tool nodes
        """
        tool_nodes = self._get_tool_nodes()

        if not tool_nodes:
            raise ValueError("Cannot ensure connected workflow: No tool nodes found in graph")

        # Check if START is already connected to something
        start_children = self.edges_map.get(START, [])

        # Check if any node is already connected to END
        nodes_to_end = []
        for node_name, children in self.edges_map.items():
            if END in children:
                nodes_to_end.append(node_name)

        if len(tool_nodes) == 1:
            # Single node workflow
            node = tool_nodes[0]
            if node.name not in start_children:
                self.add_edge(START, node.name)
            if node.name not in nodes_to_end:
                self.add_edge(node.name, END)
        else:
            # Multiple nodes - create linear chain if not already connected
            if not start_children:
                # START is not connected to anything, connect to first tool node
                self.add_edge(START, tool_nodes[0].name)

            if not nodes_to_end:
                # No node is connected to END, connect last tool node to END
                self.add_edge(tool_nodes[-1].name, END)

            # Connect tool nodes in sequence if they're not already connected
            for i in range(len(tool_nodes) - 1):
                current_node = tool_nodes[i]
                next_node = tool_nodes[i + 1]

                current_children = self.edges_map.get(current_node.name, [])
                if next_node.name not in current_children:
                    self.add_edge(current_node.name, next_node.name)

    def create_linear_tool_flow(
        self,
        tool_node_names: List[str],
        tools: List[List[Callable]],
        llm_client: Any,
        options: Optional[List[ToolLoopOptions]] = None,
        connect_to_start_end: bool = True,
    ) -> List[ToolNode]:
        """
        Create a linear flow of tool nodes.

        Args:
            tool_node_names: List of names for tool nodes
            tools: List of tool lists, one for each node
            llm_client: LLM client instance for provider API calls
            options: Optional list of options for each node (if None, uses defaults)
            connect_to_start_end: Whether to connect to START and END

        Returns:
            List of created ToolNodes
        """
        if len(tool_node_names) != len(tools):
            raise ValueError("Number of node names must match number of tool lists")

        # Normalize options list
        if options is None:
            options = [None] * len(tool_node_names)
        elif len(options) != len(tool_node_names):
            raise ValueError("If provided, options list must match number of node names")

        nodes = []
        for name, node_tools, node_options in zip(tool_node_names, tools, options):
            node = self.add_tool_node(name=name, tools=node_tools, llm_client=llm_client, options=node_options)
            nodes.append(node)

        # Connect nodes linearly
        for i in range(len(nodes) - 1):
            self.add_edge(nodes[i].name, nodes[i + 1].name)

        # Connect to START and END if requested
        if connect_to_start_end:
            if nodes:
                self.add_edge(START, nodes[0].name)
                self.add_edge(nodes[-1].name, END)

        return nodes

    def load_from_checkpoint(self, chain_id: str, checkpoint_id: Optional[str] = None) -> None:
        """
        Load graph state and execution variables from a checkpoint.
        Overrides the base Graph implementation to handle ToolState-specific attributes.

        Args:
            chain_id: The chain ID to load
            checkpoint_id: Optional specific checkpoint ID to load. If None, loads the last checkpoint.
        """
        if not self.checkpoint_storage:
            raise ValueError("Checkpoint storage must be configured to load from checkpoint")

        # Get checkpoint ID if not specified
        if not checkpoint_id:
            checkpoint_id = self.checkpoint_storage.get_last_checkpoint_id(chain_id)
            if not checkpoint_id:
                raise ValueError(f"No checkpoints found for chain {chain_id}")

        # Load checkpoint data
        if self.state:
            checkpoint = self.checkpoint_storage.load_checkpoint(
                state_instance=self.state,
                chain_id=chain_id,
                checkpoint_id=checkpoint_id,
            )

        # Verify state class matches
        current_state_class = f"{self.state.__class__.__module__}.{self.state.__class__.__name__}"
        if current_state_class != checkpoint.state_class:
            raise ValueError(
                f"State class mismatch. Current: {current_state_class}, Checkpoint: {checkpoint.state_class}"
            )

        self._clean_graph_variables()

        # Create a new state instance from serialized data using Pydantic's validation
        if self.state_class:  # Use self.state_class instead of self.initial_state
            try:
                print(
                    f"[ToolGraph.load_from_checkpoint] Attempting to validate "
                    f"JSON data for state: {checkpoint.data[:200]}..."
                )
                # --- DEBUG: Print the full JSON data being parsed ---
                print(f"[ToolGraph.load_from_checkpoint] Full JSON data:\n{checkpoint.data}")
                # --- END DEBUG ---
                # Rely on Pydantic to reconstruct the entire state, including nested models
                # Use self.state_class which should hold the correct state type
                self.state = self.state_class.model_validate_json(checkpoint.data)
                print(
                    f"[ToolGraph.load_from_checkpoint] State successfully recreated "
                    f"via model_validate_json. Type: {type(self.state)}"
                )
                # Add specific check for metadata if it exists
                if hasattr(self.state, "metadata"):
                    print(
                        f"[ToolGraph.load_from_checkpoint] Metadata type after validation: "
                        f"{type(getattr(self.state, 'metadata', None))}"
                    )

                # Store original state data for direct access during engine load_full_state
                # This might be needed if engine relies on the raw dict still
                if isinstance(checkpoint.engine_state, dict):
                    checkpoint.engine_state["data"] = checkpoint.data

            except Exception as e:
                # If model_validate_json fails, log the error and re-raise it
                # This makes the failure explicit during testing
                print(
                    f"[ToolGraph.load_from_checkpoint] CRITICAL: Error recreating state via "
                    f"model_validate_json: {str(e)}"
                )
                print(f"[ToolGraph.load_from_checkpoint] Failing data: {checkpoint.data}")
                raise  # Re-raise the exception to halt execution and see the traceback

        # Update buffers with current state values
        for field_name, buffer in self.buffers.items():
            if hasattr(self.state, field_name):
                buffer.set_value(getattr(self.state, field_name))

        # Update execution variables
        if checkpoint:
            self.chain_id = checkpoint.chain_id
            self.chain_status = checkpoint.chain_status

            if checkpoint.engine_state:
                # This will properly handle ToolState attributes
                self.execution_engine.load_full_state(checkpoint.engine_state)
            else:
                self.logger.warning("No engine state found in checkpoint")

        if self.verbose:
            self.logger.debug(f"Loaded checkpoint {checkpoint_id} for chain {self.chain_id}")

    # REMOVE _discover_custom_models and _convert_state_complex_types methods
    # Pydantic's model_validate_json should handle type reconstruction.

    def _save_checkpoint(self, node_name: str, engine_state: Dict[str, Any]) -> None:
        """
        Override the base Graph's _save_checkpoint method to ensure tool-specific state is preserved.

        Args:
            node_name: The node name where the checkpoint is being created
            engine_state: The engine state to save
        """
        if self.state and self.checkpoint_storage:
            # Create a custom checkpoint data object with ToolState-specific attributes
            checkpoint_data = CheckpointData(
                chain_id=self.chain_id,
                chain_status=self.chain_status,
                engine_state=engine_state,
            )

            # Add debug output showing what's being saved
            print(f"[ToolGraph._save_checkpoint] Saving checkpoint at node {node_name}")
            if hasattr(self.state, "is_paused"):
                print(f"[ToolGraph._save_checkpoint] State is_paused: {self.state.is_paused}")

            # Save the checkpoint
            self.checkpoint_storage.save_checkpoint(
                state_instance=self.state,
                checkpoint_data=checkpoint_data,
            )

        if self.verbose:
            self.logger.debug(f"Checkpoint saved after node: {node_name}")

    async def execute(
        self,
        timeout: Union[int, None] = None,
        chain_id: Optional[str] = None,
        auto_connect: bool = True,
    ) -> str:
        """
        Start a new execution of the tool graph.

        Args:
            timeout: Optional timeout for the execution.
            chain_id: Optional chain ID for the execution.
            auto_connect: Whether to automatically ensure workflow connectivity before execution.

        Returns:
            The chain ID of the execution.
        """
        if chain_id:
            self.chain_id = chain_id

        if not timeout:
            timeout = self.execution_timeout

        # Ensure the workflow is properly connected before execution
        if auto_connect:
            try:
                self.ensure_connected_workflow()
            except ValueError as e:
                if "No tool nodes found" in str(e):
                    raise ValueError("Cannot execute ToolGraph: No tool nodes have been added to the graph")
                raise

        self._clean_graph_variables()

        # Always use a fresh ToolEngine for a new execution
        self.execution_engine = ToolEngine(graph=self, timeout=timeout, max_node_iterations=self.max_node_iterations)

        # Execute with the graph's state
        result = await self.execution_engine.execute()

        # Update the graph's state with the execution result state
        self.state = result.state
        self.final_output = self.execution_engine.final_output

        return self.chain_id

    async def resume(
        self,
        execute_tool: bool = True,
    ) -> str:
        """
        Resume execution from a paused state.

        Args:
            execute_tool: Whether to execute the paused tool (True) or skip it (False)
            timeout: Optional timeout for the execution
            max_node_iterations: Optional maximum iterations per node

        Returns:
            The chain ID of the execution
        """
        if not self.state or not hasattr(self.state, "is_paused") or not self.state.is_paused:
            raise ValueError("Cannot resume: Graph state is not paused")

        result = await self.execution_engine.resume(self.state, execute_tool)

        # Update the graph's state with the execution result state
        self.state = result.state

        return self.chain_id


class ToolEngine(Engine):
    """
    Specialized engine for executing tool-based LLM workflows.

    This engine extends the base Engine with functionality specific to
    tool nodes, including handling LLM interactions and tool execution.
    The engine handles tool node execution differently from standard nodes,
    providing special logic for the LLM interaction loop, maintaining conversation
    history, and executing tools called by the LLM.

    Key features:
    - Manages the LLM interaction loop with tool calls
    - Tracks conversation and tool execution history
    - Handles tool errors and retries
    - Preserves checkpoint and state management from the base engine
    - Enables seamless mixing of tool nodes with standard nodes

    While this engine is designed to be used through ToolGraph's execute() and resume() methods,
    it maintains backward compatibility for direct usage in existing code.

    This engine should be used for any graph containing ToolNode instances.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_output = None

    async def resume(self, state: GraphState, execute_tool: bool = True) -> Any:
        """
        Resume execution from a paused state.

        This method is used when a tool has paused execution and the user wants to resume.

        Args:
            state: The state with pause information
            execute_tool: Whether to execute the paused tool (True) or skip it (False)
                          For pause_before_execution, this controls tool execution
                          For pause_after_execution, this controls whether to continue or not

        Returns:
            Result of execution
        """
        print("\n[ToolEngine.resume] Resuming from paused state")
        print(f"[ToolEngine.resume] Paused tool: {state.paused_tool_name}")

        if not hasattr(state, "is_paused") or not state.is_paused:
            raise ValueError("Cannot resume: State is not paused")

        if not hasattr(state, "paused_tool_name") or not state.paused_tool_name:
            raise ValueError("Cannot resume: Missing paused tool information")

        # Only needed for state consistency - ensure our engine has the current state
        self.graph.state = state

        # Find the correct tool node in the graph
        tool_node = None
        tool_node_name = None
        for node_name, node in self.graph.nodes.items():
            if hasattr(node, "is_tool_node") and node.is_tool_node:
                tool_node = node
                tool_node_name = node_name
                break

        if not tool_node:
            raise ValueError("Cannot find any tool node in the graph to resume execution")

        # Ensure tool_calls and messages are properly converted to objects, not dictionaries
        if hasattr(state, "tool_calls") and isinstance(state.tool_calls, list):
            fixed_tool_calls = []
            for tc in state.tool_calls:
                if isinstance(tc, dict) and "id" in tc and "tool_name" in tc:
                    try:
                        fixed_tool_calls.append(ToolCallLog(**tc))
                    except Exception as e:
                        print(f"[ToolEngine.resume] Error converting tool call: {str(e)}")
                        fixed_tool_calls.append(tc)  # Keep original if conversion fails
                else:
                    fixed_tool_calls.append(tc)
            state.tool_calls = fixed_tool_calls

        if hasattr(state, "messages") and isinstance(state.messages, list):
            fixed_messages = []
            for msg in state.messages:
                if isinstance(msg, dict) and "role" in msg:
                    try:
                        fixed_messages.append(LLMMessage(**msg))
                    except Exception as e:
                        print(f"[ToolEngine.resume] Error converting message: {str(e)}")
                        fixed_messages.append(msg)  # Keep original if conversion fails
                else:
                    fixed_messages.append(msg)
            state.messages = fixed_messages

        # Execute the tool if requested
        if execute_tool:
            if state.paused_after_execution:
                # For pause_after_execution, we've already executed the tool
                # and just need to continue with the next step
                print(f"[ToolEngine.resume] Tool {state.paused_tool_name} already executed (pause_after_execution)")
                print("[ToolEngine.resume] Continuing with next step...")

                # Clear the pause state
                state.is_paused = False
                state.paused_tool_id = None
                state.paused_tool_name = None
                state.paused_tool_arguments = None
                state.paused_after_execution = False
                # Keep the paused_tool_result for reference

                # Add a system message about continuing
                if hasattr(state, "messages"):
                    state.messages.append(
                        LLMMessage(
                            role="system",
                            content="Tool execution approved. Continuing with the next steps.",
                            should_show_to_user=False,
                        )
                    )
            else:
                # For pause_before_execution, we need to execute the tool
                print(f"[ToolEngine.resume] Executing tool: {state.paused_tool_name}")

                # Find the tool
                tool_func = tool_node.find_tool_by_name(state.paused_tool_name)
                if not tool_func:
                    raise ValueError(f"Tool {state.paused_tool_name} not found")

                try:
                    # Execute the tool
                    tool_id = state.paused_tool_id or f"manual_execution_{int(time.time())}"
                    tool_result = await tool_node.execute_tool(tool_func, state.paused_tool_arguments, tool_id, state)

                    # Add the result to tool calls
                    if hasattr(state, "tool_calls"):
                        # Just append the tool result to the existing list
                        if isinstance(state.tool_calls, list):
                            state.tool_calls.append(tool_result)
                        else:
                            # If not a list, initialize a new list
                            state.tool_calls = [tool_result]

                    # Add a tool result message
                    if hasattr(state, "messages"):
                        # Convert the result to a string
                        result_str = str(tool_result.result)

                        # Create a tool message
                        # Add tool result as user message instead of tool message
                        # This avoids OpenAI's requirement for tool messages to follow
                        # assistant messages with tool_calls
                        tool_message = LLMMessage(
                            role="user",
                            content=f"Tool '{state.paused_tool_name}' result: {result_str}",
                            should_show_to_user=False,
                        )

                        # Add tool result to messages and tool call entries
                        state.messages.append(tool_message)

                    print("[ToolEngine.resume] Tool execution successful")
                except Exception as e:
                    print(f"[ToolEngine.resume] Tool execution failed: {str(e)}")
                    # Add an error message
                    if hasattr(state, "messages"):
                        state.messages.append(
                            LLMMessage(
                                role="system",
                                content=f"Error executing tool {state.paused_tool_name}: {str(e)}",
                                should_show_to_user=False,
                            )
                        )
                    if hasattr(state, "error"):
                        state.error = str(e)

                # Clear the pause state
                state.is_paused = False
                state.paused_tool_id = None
                state.paused_tool_name = None
                state.paused_tool_arguments = None
                state.paused_after_execution = False
                state.paused_tool_result = None
        else:
            # Skip tool execution
            print("[ToolEngine.resume] Skipping tool execution")

            # Add a system message about skipping
            if hasattr(state, "messages"):
                state.messages.append(
                    LLMMessage(
                        role="system",
                        content=f"Tool execution skipped: {state.paused_tool_name}",
                        should_show_to_user=False,
                    )
                )

            # Clear the pause state
            state.is_paused = False
            state.paused_tool_id = None
            state.paused_tool_name = None
            state.paused_tool_arguments = None
            state.paused_after_execution = False
            state.paused_tool_result = None

        # Continue execution with the identified tool node
        print(f"[ToolEngine.resume] Continuing with execution of node: {tool_node_name}")
        self.graph._update_chain_status(ChainStatus.RUNNING)

        # Execute the node loop
        try:
            # Create a frame for the tool node
            frame = ExecutionFrame(tool_node_name, state)

            # Execute the tool node directly
            print(f"[ToolEngine.resume] Executing tool node: {tool_node_name}")
            await self._execute_tool_node(frame)

            # Mark as complete after tool execution
            if hasattr(state, "is_complete"):
                state.is_complete = True

            # Get the final output if available
            if hasattr(state, "final_output") and not state.final_output:
                if hasattr(state, "messages") and state.messages:
                    # Get the last assistant message as final output
                    for msg in reversed(state.messages):
                        # Handle both LLMMessage objects and dictionaries
                        if isinstance(msg, dict):
                            if msg.get("role") == "assistant" and msg.get("content"):
                                state.final_output = msg.get("content")
                                break
                        elif hasattr(msg, "role") and hasattr(msg, "content"):
                            if msg.role == "assistant" and msg.content:
                                state.final_output = msg.content
                                break

            print("[ToolEngine.resume] Execution completed successfully")
        except Exception as e:
            error = str(e)
            print(f"[ToolEngine.resume] Error in node execution: {error}")
            if hasattr(state, "error"):
                state.error = error

            # Update chain status to FAILED on error
            self.graph._update_chain_status(ChainStatus.FAILED)

        # Return the result
        class ExecutionResult:
            def __init__(self, state):
                self.state = state

        return ExecutionResult(state)

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get the complete executor state for checkpointing/resumption.
        Extends the parent method to include ToolState attributes.
        """
        # First, ensure the pause state is correctly set in the state object
        # before getting the full state. This is critical for correct checkpointing.
        if hasattr(self.graph, "state") and isinstance(self.graph.state, ToolState):
            print(f"[ToolEngine.get_full_state] State object before saving: is_paused={self.graph.state.is_paused}")
            print(f"[ToolEngine.get_full_state] Paused tool name: {self.graph.state.paused_tool_name}")

        # Get the base engine state from parent method
        state = super().get_full_state()

        # Add explicit debug logging for better understanding during testing
        print(f"[ToolEngine.get_full_state] Saving state, chain_status: {self.graph.chain_status}")

        # Add ToolState-specific attributes if available
        if hasattr(self.graph, "state") and isinstance(self.graph.state, ToolState):
            # Add pause state attributes - use graph state values directly
            is_paused = self.graph.state.is_paused
            print(f"[ToolEngine.get_full_state] Saving is_paused: {is_paused}")

            # Make sure current is_paused value is captured
            state["is_paused"] = is_paused
            state["paused_tool_id"] = self.graph.state.paused_tool_id
            state["paused_tool_name"] = self.graph.state.paused_tool_name
            state["paused_tool_arguments"] = self.graph.state.paused_tool_arguments
            state["paused_after_execution"] = self.graph.state.paused_after_execution
            state["paused_tool_result"] = self.graph.state.paused_tool_result

            # Also store them in a structured format for clarity in future checkpoint format
            pause_state = {
                "is_paused": is_paused,
                "paused_tool_id": self.graph.state.paused_tool_id,
                "paused_tool_name": self.graph.state.paused_tool_name,
                "paused_tool_arguments": self.graph.state.paused_tool_arguments,
                "paused_after_execution": self.graph.state.paused_after_execution,
                "paused_tool_result": self.graph.state.paused_tool_result,
            }
            state["graph_pause_state"] = pause_state

            # Store message and tool call history
            if hasattr(self.graph.state, "messages"):
                state["tool_state_messages"] = self.graph.state.messages
                print(f"[ToolEngine.get_full_state] Saving {len(self.graph.state.messages)} messages")
            if hasattr(self.graph.state, "tool_calls"):
                state["tool_state_tool_calls"] = self.graph.state.tool_calls
                print(f"[ToolEngine.get_full_state] Saving {len(self.graph.state.tool_calls)} tool calls")

            # Store additional ToolState properties
            if hasattr(self.graph.state, "current_iteration"):
                state["tool_state_current_iteration"] = self.graph.state.current_iteration
            if hasattr(self.graph.state, "max_iterations"):
                state["tool_state_max_iterations"] = self.graph.state.max_iterations
            if hasattr(self.graph.state, "is_complete"):
                state["tool_state_is_complete"] = self.graph.state.is_complete
            if hasattr(self.graph.state, "final_output"):
                state["tool_state_final_output"] = self.graph.state.final_output
            if hasattr(self.graph.state, "error"):
                state["tool_state_error"] = self.graph.state.error
            if hasattr(self.graph.state, "current_trace"):
                state["tool_state_current_trace"] = self.graph.state.current_trace
            if hasattr(self.graph.state, "raw_response_history"):
                state["tool_state_raw_response_history"] = self.graph.state.raw_response_history

            # NEW: Save all custom fields from the state
            # Get all fields from the state class
            if "model_fields" in dir(self.graph.state):
                # Pydantic v2
                state_fields = list(self.graph.state.model_fields.keys())
            else:
                # Fallback to checking all non-private, non-callable attributes
                state_fields = [
                    f
                    for f in dir(self.graph.state)
                    if not f.startswith("_") and not callable(getattr(self.graph.state, f))
                ]

            # Add custom fields to state dictionary
            for field_name in state_fields:
                # Skip fields we've already handled
                if field_name.startswith("_") or field_name in state:
                    continue

                try:
                    # Get field value
                    field_value = getattr(self.graph.state, field_name)
                    # Only save if not None and not a method
                    if field_value is not None and not callable(field_value):
                        state[field_name] = field_value
                        print(f"[ToolEngine.get_full_state] Saving custom field: {field_name}")
                except Exception as e:
                    print(f"[ToolEngine.get_full_state] Error saving field {field_name}: {str(e)}")

            # Make sure the pause state is also set in each execution frame
            if "execution_frames" in state and state["execution_frames"]:
                for frame in state["execution_frames"]:
                    if frame and hasattr(frame, "state") and frame.state and hasattr(frame.state, "is_paused"):
                        print(f"[ToolEngine.get_full_state] Found frame with is_paused: {frame.state.is_paused}")
                        if frame.state.is_paused:
                            self.graph.state.is_paused = frame.state.is_paused
                            if hasattr(frame.state, "paused_tool_id"):
                                self.graph.state.paused_tool_id = frame.state.paused_tool_id
                            if hasattr(frame.state, "paused_tool_name"):
                                self.graph.state.paused_tool_name = frame.state.paused_tool_name
                            if hasattr(frame.state, "paused_tool_arguments"):
                                self.graph.state.paused_tool_arguments = frame.state.paused_tool_arguments
                            if hasattr(frame.state, "paused_after_execution"):
                                self.graph.state.paused_after_execution = frame.state.paused_after_execution
                            if hasattr(frame.state, "paused_tool_result"):
                                self.graph.state.paused_tool_result = frame.state.paused_tool_result
                            break

            if self.graph.state.is_paused:
                print(
                    f"[ToolEngine.get_full_state] Restored pause state: paused={self.graph.state.is_paused}, "
                    f"tool={self.graph.state.paused_tool_name}"
                )
                # Ensure chain status is set to PAUSE
                self.graph._update_chain_status(ChainStatus.PAUSE)
            else:
                print("[ToolEngine.get_full_state] Restored tool state (not paused)")

        return state

    def load_full_state(self, saved_state: Dict[str, Any]) -> None:
        """
        Restore the complete executor state from a saved snapshot.
        Extends the parent Engine method to properly handle ToolState pause attributes.
        """
        # Call parent method to load the basic engine state
        super().load_full_state(saved_state)

        # Debug log the saved state keys
        pause_keys = [k for k in saved_state.keys() if "pause" in k.lower()]
        tool_keys = [k for k in saved_state.keys() if "tool_state" in k]
        print(f"[ToolEngine.load_full_state] Found pause-related keys: {pause_keys}")
        print(f"[ToolEngine.load_full_state] Found tool state keys: {tool_keys}")

        # Additionally, restore the ToolState-specific attributes if they exist in the state
        if hasattr(self.graph, "state") and isinstance(self.graph.state, ToolState):
            # Check for pause-related attributes directly in the saved state
            # This handles existing checkpoints that store this info at the top level
            if hasattr(self.graph.state, "is_paused") and "is_paused" in saved_state:
                print(f"[ToolEngine.load_full_state] Setting is_paused from saved_state: {saved_state['is_paused']}")
                self.graph.state.is_paused = saved_state["is_paused"]
            if hasattr(self.graph.state, "paused_tool_id") and "paused_tool_id" in saved_state:
                self.graph.state.paused_tool_id = saved_state["paused_tool_id"]
            if hasattr(self.graph.state, "paused_tool_name") and "paused_tool_name" in saved_state:
                self.graph.state.paused_tool_name = saved_state["paused_tool_name"]
            if hasattr(self.graph.state, "paused_tool_arguments") and "paused_tool_arguments" in saved_state:
                self.graph.state.paused_tool_arguments = saved_state["paused_tool_arguments"]
            if hasattr(self.graph.state, "paused_after_execution") and "paused_after_execution" in saved_state:
                self.graph.state.paused_after_execution = saved_state["paused_after_execution"]
            if hasattr(self.graph.state, "paused_tool_result") and "paused_tool_result" in saved_state:
                # Convert dict to ToolCallLog if needed
                if (
                    isinstance(saved_state["paused_tool_result"], dict)
                    and "id" in saved_state["paused_tool_result"]
                    and "tool_name" in saved_state["paused_tool_result"]
                ):
                    try:
                        self.graph.state.paused_tool_result = ToolCallLog(**saved_state["paused_tool_result"])
                    except Exception as e:
                        print(f"[ToolEngine.load_full_state] Error converting paused_tool_result: {str(e)}")
                        self.graph.state.paused_tool_result = saved_state["paused_tool_result"]
                else:
                    self.graph.state.paused_tool_result = saved_state["paused_tool_result"]

            # Check for pause-related attributes in graph_pause_state structure
            # This handles the new format introduced with this update
            if "graph_pause_state" in saved_state:
                pause_state = saved_state["graph_pause_state"]
                print(f"[ToolEngine.load_full_state] Found graph_pause_state: {pause_state}")

                # Set the pause attributes on the graph's state
                if hasattr(self.graph.state, "is_paused") and "is_paused" in pause_state:
                    print(
                        f"[ToolEngine.load_full_state] Setting is_paused from pause_state: {pause_state['is_paused']}"
                    )
                    self.graph.state.is_paused = pause_state["is_paused"]
                if hasattr(self.graph.state, "paused_tool_id") and "paused_tool_id" in pause_state:
                    self.graph.state.paused_tool_id = pause_state["paused_tool_id"]
                if hasattr(self.graph.state, "paused_tool_name") and "paused_tool_name" in pause_state:
                    self.graph.state.paused_tool_name = pause_state["paused_tool_name"]
                if hasattr(self.graph.state, "paused_tool_arguments") and "paused_tool_arguments" in pause_state:
                    self.graph.state.paused_tool_arguments = pause_state["paused_tool_arguments"]
                if hasattr(self.graph.state, "paused_after_execution") and "paused_after_execution" in pause_state:
                    self.graph.state.paused_after_execution = pause_state["paused_after_execution"]
                if hasattr(self.graph.state, "paused_tool_result") and "paused_tool_result" in pause_state:
                    # Convert dict to ToolCallLog if needed
                    if (
                        isinstance(pause_state["paused_tool_result"], dict)
                        and "id" in pause_state["paused_tool_result"]
                        and "tool_name" in pause_state["paused_tool_result"]
                    ):
                        try:
                            self.graph.state.paused_tool_result = ToolCallLog(**pause_state["paused_tool_result"])
                        except Exception as e:
                            print(
                                f"[ToolEngine.load_full_state] Error converting \
                                    paused_tool_result from pause_state: {str(e)}"
                            )
                            self.graph.state.paused_tool_result = pause_state["paused_tool_result"]
                    else:
                        self.graph.state.paused_tool_result = pause_state["paused_tool_result"]

            # Restore tool_calls and messages history
            if hasattr(self.graph.state, "tool_calls") and "tool_state_tool_calls" in saved_state:
                print(
                    f"[ToolEngine.load_full_state] Restoring tool_calls from saved_state, \
                        found {len(saved_state['tool_state_tool_calls'])} entries"
                )
                # Process each tool call to ensure it's a ToolCallLog object
                tool_calls = []
                for item in saved_state["tool_state_tool_calls"]:
                    if isinstance(item, ToolCallLog):
                        tool_calls.append(item)
                    elif isinstance(item, dict) and "id" in item and "tool_name" in item:
                        try:
                            # Convert dict to ToolCallLog
                            tool_calls.append(ToolCallLog(**item))
                        except Exception as e:
                            print(f"[ToolEngine.load_full_state] Error converting tool call: {str(e)}")
                            # If conversion fails, still include the original item
                            tool_calls.append(item)
                    else:
                        # If it's not a valid format, include as is
                        tool_calls.append(item)

                self.graph.state.tool_calls = tool_calls
            if hasattr(self.graph.state, "messages") and "tool_state_messages" in saved_state:
                print(
                    f"[ToolEngine.load_full_state] Restoring messages from saved_state, \
                        found {len(saved_state['tool_state_messages'])} entries"
                )
                # Process each message to ensure it's an LLMMessage object
                messages = []
                for item in saved_state["tool_state_messages"]:
                    if isinstance(item, LLMMessage):
                        messages.append(item)
                    elif isinstance(item, dict) and "role" in item:
                        try:
                            # Convert dict to LLMMessage
                            messages.append(LLMMessage(**item))
                        except Exception as e:
                            print(f"[ToolEngine.load_full_state] Error converting message: {str(e)}")
                            # If conversion fails, still include the original item
                            messages.append(item)
                    else:
                        # If it's not a valid format, include as is
                        messages.append(item)

                self.graph.state.messages = messages
                print(
                    f"[ToolEngine.load_full_state] First message type after conversion: "
                    f"{type(messages[0]) if messages else 'None'}"
                )

            # CRITICAL: Handle direct state messages field directly from the state
            if hasattr(self.graph.state, "messages") and isinstance(self.graph.state.messages, list):
                messages = self.graph.state.messages
                if messages and isinstance(messages[0], dict) and "role" in messages[0]:
                    print("[ToolEngine.load_full_state] Converting messages directly in state object")
                    fixed_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and "role" in msg:
                            try:
                                fixed_messages.append(LLMMessage(**msg))
                            except Exception as e:
                                print(f"[ToolEngine.load_full_state] Error converting message: {str(e)}")
                                fixed_messages.append(msg)
                        else:
                            fixed_messages.append(msg)
                    self.graph.state.messages = fixed_messages

            # Apply message conversion from any serialized data
            try:
                if isinstance(saved_state, dict) and "data" in saved_state and saved_state["data"]:
                    import json

                    state_dict = json.loads(saved_state["data"])
                    if "messages" in state_dict and isinstance(state_dict["messages"], list):
                        messages = state_dict["messages"]
                        print(f"[ToolEngine.load_full_state] Found messages in state data: {len(messages)}")
                        if messages and isinstance(messages[0], dict) and "role" in messages[0]:
                            fixed_messages = []
                            for msg in messages:
                                if isinstance(msg, dict) and "role" in msg:
                                    try:
                                        fixed_messages.append(LLMMessage(**msg))
                                    except Exception as e:
                                        print(
                                            f"[ToolEngine.load_full_state] Error converting message from data: {str(e)}"
                                        )
                                        fixed_messages.append(msg)
                                else:
                                    fixed_messages.append(msg)
                            self.graph.state.messages = fixed_messages
            except Exception as e:
                print(f"[ToolEngine.load_full_state] Error processing state data: {str(e)}")

            # Restore additional ToolState properties
            if hasattr(self.graph.state, "current_iteration") and "tool_state_current_iteration" in saved_state:
                self.graph.state.current_iteration = saved_state["tool_state_current_iteration"]
            if hasattr(self.graph.state, "max_iterations") and "tool_state_max_iterations" in saved_state:
                self.graph.state.max_iterations = saved_state["tool_state_max_iterations"]
            if hasattr(self.graph.state, "is_complete") and "tool_state_is_complete" in saved_state:
                self.graph.state.is_complete = saved_state["tool_state_is_complete"]
            if hasattr(self.graph.state, "final_output") and "tool_state_final_output" in saved_state:
                self.graph.state.final_output = saved_state["tool_state_final_output"]
            if hasattr(self.graph.state, "error") and "tool_state_error" in saved_state:
                self.graph.state.error = saved_state["tool_state_error"]
            if hasattr(self.graph.state, "current_trace") and "tool_state_current_trace" in saved_state:
                self.graph.state.current_trace = saved_state["tool_state_current_trace"]
            if hasattr(self.graph.state, "raw_response_history") and "tool_state_raw_response_history" in saved_state:
                self.graph.state.raw_response_history = saved_state["tool_state_raw_response_history"]

            # REMOVED: Section that manually checked/restored custom fields using setattr.
            # The self.graph.state object is assumed to be correctly reconstructed by
            # ToolGraph.load_from_checkpoint using model_validate_json.
            # We only need to restore engine-specific state here.

            # Restore pause state information from saved_state if necessary
            # (Ideally, pause state should also be part of the GraphState model)
            if self.graph.state.is_paused:
                print(
                    f"[ToolEngine.load_full_state] Restored pause state from GraphState: "
                    f"paused={self.graph.state.is_paused}, "
                    f"tool={self.graph.state.paused_tool_name}"
                )
                # Ensure chain status is set to PAUSE
                self.graph._update_chain_status(ChainStatus.PAUSE)
            else:
                print("[ToolEngine.load_full_state] Restored tool state (not paused) from GraphState")

    async def execute(self) -> Any:
        """
        Begin executing the graph from the START node using graph's state.

        Returns:
            Result of execution
        """
        print("\n[ToolEngine.execute] Starting execution with graph's state")

        self._has_executed = True  # Mark that execute() has been run

        # Always use the graph's state
        state_to_use = self.graph.state

        print(f"[ToolEngine.execute] Using state: {type(state_to_use)}")
        print(f"[ToolEngine.execute] Graph nodes: {list(self.graph.nodes.keys())}")
        print(f"[ToolEngine.execute] Graph edges: {self.graph.edges_map}")

        try:
            # Create initial frame and queue it for execution
            initial_frame = ExecutionFrame(START, state_to_use)
            print("[ToolEngine.execute] Created initial frame with START node")
            self.execution_frames.append(initial_frame)

            # Set chain status to running
            self.graph._update_chain_status(ChainStatus.RUNNING)

            # Execute until complete or interrupted
            print("[ToolEngine.execute] Calling _execute_all")
            await self._execute_all()
            print("[ToolEngine.execute] _execute_all completed")

        except Exception as e:
            error_msg = str(e)
            print(f"[ToolEngine.execute] Error during execution: {error_msg}")
            self.graph._update_chain_status(ChainStatus.FAILED)
            raise

        print(f"[ToolEngine.execute] Returning result with state: {state_to_use}")

        # For backward compatibility with existing code that uses ToolEngine directly
        class ExecutionResult:
            def __init__(self, state):
                self.state = state

        return ExecutionResult(state_to_use)

    def _validate_message_against_schema(
        self, content: str, output_schema: Optional[Type[BaseModel]]
    ) -> tuple[bool, Optional[BaseModel], Optional[str]]:
        """
        Validate a message content against the output schema.

        Args:
            content: The message content to validate
            output_schema: The Pydantic model to validate against

        Returns:
            Tuple of (is_valid, parsed_model, error_message)
        """
        if not output_schema or not content.strip():
            return False, None, None

        try:
            # Try to parse as JSON first
            parsed_data = json.loads(content.strip())

            # Validate against schema
            validated_model = output_schema(**parsed_data)

            return True, validated_model, None

        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON format: {e}"
        except Exception as e:
            return False, None, f"Schema validation failed: {e}"

    def _handle_system_prompt(self, node: ToolNode, state: ToolState) -> ToolState:
        # Ensure system prompt is always present as the first message
        if not state.messages or state.messages[0].role != "system":
            system_message = LLMMessage(
                role="system",
                content=getattr(
                    node, "system_prompt", "You are a helpful AI assistant that can use tools to accomplish tasks."
                ),
                should_show_to_user=False,
            )
            state.messages.insert(0, system_message)

        elif state.messages[0].role == "system":
            # Update existing system message with current system prompt
            state.messages[0].content = getattr(node, "system_prompt", state.messages[0].content)

        return state

    def _infer_llm_provider(self, node: ToolNode) -> str:
        # Determine LLM provider and prepare tools
        provider = "openai"  # Default
        if (
            hasattr(node.llm_client, "client")
            and node.llm_client.client
            and hasattr(node.llm_client.client, "__class__")
        ):
            # Check if this is an Anthropic client by module name
            client_module = node.llm_client.client.__class__.__module__
            if "anthropic" in client_module:
                # Anthropic doesn't support the "auto" string
                provider = "anthropic"
        return provider

    def _set_up_streaming(self, node: ToolNode, state: ToolState) -> StreamingConfig:
        """
        Set up streaming configuration for a tool node.
        """
        # Set up streaming configuration if enabled
        streaming_config = node.options.streaming_config

        # If no streaming_config but streaming options are set, create one
        if streaming_config is None and (
            node.options.stream
            or node.options.stream_events
            or node.options.redis_host
            or node.options.redis_channel
            or node.options.stream_callback
        ):
            from primeGraph.graph.llm_clients import StreamingConfig

            streaming_config = StreamingConfig(
                enabled=node.options.stream,
                event_types=node.options.stream_events or {StreamingEventType.TEXT},
                redis_host=node.options.redis_host,
                redis_port=node.options.redis_port,
                redis_channel=node.options.redis_channel,
                callback=node.options.stream_callback,
            )

        # Set streaming state fields
        if hasattr(state, "streaming_enabled") and streaming_config:
            state.streaming_enabled = streaming_config.enabled
            if streaming_config.redis_channel:
                state.streaming_channel = streaming_config.redis_channel

        return streaming_config

    def _handle_pause_state(self, node: ToolNode, state: ToolState) -> ToolState:
        # Handle pause state, if we're resuming from a pause
        if hasattr(state, "is_paused") and state.is_paused:
            logger.info(f"Resuming from paused state, tool: {state.paused_tool_name}")
            if hasattr(state, "paused_after_execution") and state.paused_after_execution:
                logger.info("We're resuming after a tool has already been executed")
                # Handle tool result that was already executed but paused after
                # We'll skip straight to the result handling

                # Clear the pause state
                state.is_paused = False
                state.paused_tool_name = None
                state.paused_tool_id = None
                state.paused_tool_arguments = None
                state.paused_after_execution = False

                # Keep the paused_tool_result for processing
                # You might want to handle this result differently...

                # Save this state change to checkpoint
                self._capture_state_checkpoint(node.name, state)
            # If not paused after execution, we're paused before execution of a tool
            # Let the normal flow handle this

        return state

    def _convert_model_message_to_dict(self, state: ToolState) -> Dict[str, Any]:
        """
        Convert a Pydantic model message to a dictionary.
        """
        message_dicts = []
        for msg in state.messages:
            if hasattr(msg, "model_dump"):
                # Use pydantic's model_dump method if available
                message_dicts.append(msg.model_dump())
            elif hasattr(msg, "dict"):
                # For older pydantic versions
                message_dicts.append(msg.dict())
            elif isinstance(msg, dict):
                # Handle already-dictionary messages
                message_dicts.append(msg)
            else:
                # Fallback to manual conversion - handle both objects and dicts
                if isinstance(msg, dict):
                    msg_dict = {"role": msg.get("role", ""), "content": msg.get("content", "")}
                    if msg.get("tool_calls") is not None:
                        msg_dict["tool_calls"] = msg.get("tool_calls")
                    if msg.get("tool_call_id") is not None:
                        msg_dict["tool_call_id"] = msg.get("tool_call_id")
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    msg_dict = {"role": msg.role, "content": msg.content}
                    if hasattr(msg, "tool_calls") and msg.tool_calls is not None:
                        msg_dict["tool_calls"] = msg.tool_calls
                    if hasattr(msg, "tool_call_id") and msg.tool_call_id is not None:
                        msg_dict["tool_call_id"] = msg.tool_call_id
                else:
                    # Skip invalid messages
                    print(f"Warning: Skipping invalid message: {msg}")
                    continue
                message_dicts.append(msg_dict)

        logger.debug(f"Converted {len(message_dicts)} messages to dicts")
        logger.debug(f"Message dicts: {message_dicts}")
        return message_dicts

    def _store_raw_response(self, state: ToolState, response: Dict[str, Any], content: str) -> ToolState:
        if hasattr(state, "raw_response_history"):
            # Convert the response to a safely serializable format
            try:
                # Extract only the basic properties that we need and ensure they're serializable
                serializable_response = {
                    "timestamp": time.time(),
                    "provider": getattr(response, "provider", "unknown")
                    if not isinstance(response, dict)
                    else response.get("provider", "unknown"),
                    "id": getattr(response, "id", None) if not isinstance(response, dict) else response.get("id"),
                    "role": getattr(response, "role", None) if not isinstance(response, dict) else response.get("role"),
                    "model": getattr(response, "model", None)
                    if not isinstance(response, dict)
                    else response.get("model"),
                    "content_summary": content[:100] + "..." if len(content) > 100 else content,
                }

                # Add any other safely serializable properties
                if hasattr(response, "usage"):
                    try:
                        serializable_response["usage"] = (
                            response.usage._asdict() if hasattr(response.usage, "_asdict") else dict(response.usage)
                        )
                    except Exception:
                        # If usage can't be converted, store a string representation
                        serializable_response["usage"] = str(response.usage)

                state.raw_response_history.append(serializable_response)
            except Exception as e:
                # If conversion fails, store a minimal record instead of the raw response
                print(f"Error converting response for history: {str(e)}")
                state.raw_response_history.append({"timestamp": time.time(), "error": str(e)})

    def _handle_tool_choice(self, node: ToolNode, tool_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        tool_choice_param = None
        if (
            hasattr(node.llm_client, "client")
            and node.llm_client.client
            and hasattr(node.llm_client.client, "__class__")
        ):
            # Check if this is an Anthropic client by module name
            client_module = node.llm_client.client.__class__.__module__
            if "anthropic" in client_module:
                # Anthropic uses different parameters
                tool_choice_param = None
            elif "openai" in client_module:
                # For OpenAI, we need to format the tool_choice parameter differently
                tool_choice_param = {"type": "function"}
                # For OpenAI, it requires a specific format with a function parameter
                # If any tools are available, use the first one to avoid "missing tool_choice.function" error
                if tool_schemas and len(tool_schemas) > 0:
                    first_tool = tool_schemas[0]
                    if isinstance(first_tool, dict) and "function" in first_tool:
                        tool_choice_param = {
                            "type": "function",
                            "function": {"name": first_tool["function"]["name"]},
                        }
        return tool_choice_param

    def _extract_tool_calls(
        self, node: ToolNode, response: Dict[str, Any], state: ToolState, content: str
    ) -> List[Dict[str, Any]]:
        tool_calls = []
        try:
            tool_calls, tool_message = node.llm_client.extract_tool_calls(response)
            logger.debug(f"Extracted {len(tool_calls)} tool calls. Tool calls: {tool_calls}")
        except Exception as e:
            logger.error(f"Error extracting tool calls via client: {str(e)}")
            # Try manual extraction as fallback
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            # The arguments are already parsed by json.loads in the client
                            args = tc.function.arguments
                            if isinstance(args, str):
                                args = json.loads(args)
                        except Exception as e:
                            logger.error(f"Error parsing tool arguments: {str(e)}")
                            args = {"input": tc.function.arguments}

                        tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
            elif hasattr(response, "content") and isinstance(response.content, list):
                for block in response.content:
                    if getattr(block, "type", None) == "tool_use":
                        # For Anthropic, the input is already a dict
                        args = getattr(block, "input", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception as e:
                                logger.error(f"Error parsing tool arguments: {str(e)}")
                                args = {"input": args}

                        tool_calls.append(
                            {
                                "id": getattr(block, "id", f"tool_{int(time.time())}"),
                                "name": getattr(block, "name", ""),
                                "arguments": args,
                            }
                        )

        logger.debug(f"Extracted {len(tool_calls)} tool calls")

        return state, tool_calls, tool_message

    def _on_message(self, node: ToolNode, message: Dict[str, Any]) -> None:
        """
        Handle the on_message callback for a tool node.
        """
        if hasattr(node, "on_message") and node.on_message:
            node.on_message(message)

    async def _execute_tool_node(self, frame: ExecutionFrame) -> Dict[str, Any]:
        """
        Execute a tool node.

        This is a specialized execution loop for tool nodes that interact with LLMs and tools.
        It handles the conversation loop with an LLM, gathering tool calls, executing them,
        and building up the conversation history.

        Args:
            frame: The execution frame with the current node and state

        Returns:
            The updated state after execution
        """
        node_name = frame.node_id
        node = self.graph.nodes[node_name]
        state = frame.state

        # Debug info
        logger.info(f"\nExecuting tool node: {node_name}")
        logger.info(f"State type: {type(state)}")
        logger.info(
            f"State fields: {state.model_fields.keys() if hasattr(state, 'model_fields') else 'No model_fields'}"
        )

        # Get options from the node
        max_iterations = node.options.max_iterations
        if hasattr(state, "max_iterations"):
            state.max_iterations = max_iterations

        # Set up streaming configuration if enabled
        streaming_config = self._set_up_streaming(node, state)

        # Initialize tracking variables
        is_complete = False
        error = None
        buffer_updates = {}
        tool_call_entries = []

        # Ensure system prompt is always present as the first message
        state = self._handle_system_prompt(node, state)

        # IMPORTANT: Save the state to ensure initial conditions are captured
        self._capture_state_checkpoint(node_name, state)

        # Determine LLM provider and prepare tools
        provider = self._infer_llm_provider(node)

        # Generate tool schemas for the provider
        tool_schemas = node.get_tool_schemas(provider)
        logger.info(f"Generated {len(tool_schemas)} tool schemas")

        # Begin conversation loop
        current_iteration = 0
        if hasattr(state, "current_iteration"):
            current_iteration = state.current_iteration

        # Handle pause state, if we're resuming from a pause
        state = self._handle_pause_state(node, state)

        # Main tool loop
        while current_iteration < max_iterations and not is_complete and not error:
            print(f"\nTool loop iteration {current_iteration + 1}/{max_iterations}")

            # Save checkpoint at the start of each iteration
            self._capture_state_checkpoint(node_name, state)

            # Set current iteration in state
            if hasattr(state, "current_iteration"):
                state.current_iteration = current_iteration

            try:
                # Call LLM with current messages and tools

                logger.info(f"Calling LLM generate with {len(state.messages)} messages and {len(tool_schemas)} tools")
                logger.debug(f"Tool schemas: {tool_schemas}")
                logger.debug(f"State messages: {state.messages}")

                # Convert LLMMessage objects to dictionaries
                message_dicts = self._convert_model_message_to_dict(state)

                # Get API kwargs, ensuring model is included if specified in options
                api_kwargs = node.options.api_kwargs.copy()
                if node.options.model:
                    api_kwargs["model"] = node.options.model

                # Use max_tokens from options
                if node.options.max_tokens and "max_tokens" not in api_kwargs:
                    api_kwargs["max_tokens"] = node.options.max_tokens

                # Make the API call with proper parameters
                content, response = await node.llm_client.generate(
                    messages=message_dicts,
                    tools=tool_schemas,
                    streaming_config=streaming_config,
                    **api_kwargs,
                )

                logger.debug(f"LLM raw response: {response}")

                # Update last stream timestamp if streaming was enabled
                if streaming_config and streaming_config.enabled and hasattr(state, "last_stream_timestamp"):
                    state.last_stream_timestamp = time.time()

                # Check if this is a tool use response
                is_tool_use = False
                try:
                    is_tool_use = node.llm_client.is_tool_use_response(response)
                except Exception as e:
                    logger.error(f"Error checking for tool use: {str(e)}")
                    # Continue with default (False)

                logger.info(f"is_tool_use_response: {is_tool_use}")

                # Store raw response for debugging/logging if needed
                self._store_raw_response(state, response, content)

                if is_tool_use:
                    state, tool_calls, tool_message = self._extract_tool_calls(node, response, state, content)
                    state.messages.append(tool_message)

                    # If there are no tool calls but is_tool_use_response was True,
                    # this is likely an empty tool_calls list, which should be treated as a non-tool response
                    if not tool_calls:
                        logger.debug("No tool calls found in response, treating as non-tool response")
                        # Handle this as a non-tool response by falling through to the final response handling
                        is_tool_use = False

                    # Process tool calls in parallel
                    logger.info(f"Processing {len(tool_calls)} tool calls in parallel")

                    # First, validate all tool calls and check for pause_before_execution
                    valid_tool_calls = []
                    pause_before_tools = []

                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_id = tc.get("id", f"call_{int(time.time())}")
                        arguments = tc.get("arguments", {})

                        logger.info(f"Validating tool call: {tool_name}({arguments})")

                        # Find the tool by name
                        tool_func = node.find_tool_by_name(tool_name)
                        if not tool_func:
                            error_msg = f"Tool {tool_name} not found"
                            tool_error_message = LLMMessage(
                                role="system", content=f"Error: {error_msg}", should_show_to_user=False
                            )
                            state.messages.append(tool_error_message)

                            # Call the on_message callback for the error message
                            self._on_message(
                                node,
                                {
                                    "message_type": "system",
                                    "content": f"Error: {error_msg}",
                                    "is_final": False,
                                    "is_error": True,
                                    "iteration": current_iteration,
                                    "timestamp": time.time(),
                                },
                            )
                            continue  # Skip invalid tool

                        # Check for pause_before_execution
                        if hasattr(tool_func, "_tool_definition") and tool_func._tool_definition.pause_before_execution:
                            pause_before_tools.append((tool_func, tool_name, tool_id, arguments))

                        valid_tool_calls.append((tool_func, tool_name, tool_id, arguments))

                    # If any tool requires pause_before_execution, handle the first one
                    if pause_before_tools:
                        tool_func, tool_name, tool_id, arguments = pause_before_tools[0]
                        logger.info(f"Pausing execution before tool: {tool_name}")

                        state.is_paused = True
                        state.paused_tool_id = tool_id
                        state.paused_tool_name = tool_name
                        state.paused_tool_arguments = arguments
                        state.paused_after_execution = False
                        state.paused_tool_result = None

                        # Update chain status to PAUSE
                        self.graph._update_chain_status(ChainStatus.PAUSE)

                        # CRITICAL: Save pause state to checkpoint
                        self._capture_state_checkpoint(node_name, state)

                        # Return early with updated state
                        if hasattr(state, "is_complete"):
                            state.is_complete = False
                        return {"state": state}

                    if not valid_tool_calls:
                        logger.debug("No valid tool calls to execute")
                        continue

                    # Execute all valid tool calls in parallel
                    async def execute_single_tool(tool_data):
                        tool_func, tool_name, tool_id, arguments = tool_data
                        logger.info(f"Executing tool: {tool_name}")

                        try:
                            # Execute the tool and capture the result
                            tool_result, tool_message = await node.execute_tool(tool_func, arguments, tool_id, state)

                            return tool_result, tool_message, tool_func, tool_name, tool_id, arguments
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {str(e)}")
                            # Create a failed tool result
                            failed_result = ToolCallLog(
                                id=tool_id,
                                tool_name=tool_name,
                                arguments=arguments,
                                result=None,
                                success=False,
                                timestamp=time.time(),
                                duration_ms=0.0,
                                error=str(e),
                            )
                            return failed_result, None, tool_func, tool_name, tool_id, arguments

                    # Execute all tools in parallel
                    logger.info(f"Executing {len(valid_tool_calls)} tools in parallel")

                    # TODO: Add a force break when end_conversation is triggered
                    parallel_results = await asyncio.gather(
                        *[execute_single_tool(tool_data) for tool_data in valid_tool_calls], return_exceptions=True
                    )

                    # Process results
                    pause_after_tools = []
                    abort_after_tools = []

                    for result in parallel_results:
                        if isinstance(result, Exception):
                            logger.error(f"Tool execution failed with exception: {result}")
                            continue

                        tool_result, tool_message, tool_func, tool_name, tool_id, arguments = result
                        logger.info(f"Tool result: {tool_result}")

                        # Add the result to tool calls
                        if hasattr(state, "tool_calls"):
                            if isinstance(state.tool_calls, list):
                                state.tool_calls.append(tool_result)
                            else:
                                state.tool_calls = [tool_result]

                        # Check for pause_after_execution
                        if hasattr(tool_func, "_tool_definition") and tool_func._tool_definition.pause_after_execution:
                            pause_after_tools.append((tool_result, tool_func, tool_name, tool_id, arguments))

                        # Check for abort_after_execution
                        if hasattr(tool_func, "_tool_definition") and tool_func._tool_definition.abort_after_execution:
                            abort_after_tools.append((tool_result, tool_func, tool_name, tool_id, arguments))

                        # Add tool result to tool_call_entries
                        tool_call_entries.append(tool_result)

                        if tool_message:
                            # Add tool result to messages
                            state.messages.append(tool_message)

                            # Call the on_message callback for the tool result message
                            self._on_message(node, tool_message)

                        if tool_name == "end_conversation":
                            logger.info("End conversation triggered, breaking out of tool loop")
                            is_complete = True

                            logger.info(f"Node output schema: {node.output_schema}")

                            buffer_updates["is_complete"] = True
                            buffer_updates["final_output"] = node.output_schema(**json.loads(tool_result.result))
                            self.final_output = node.output_schema(**json.loads(tool_result.result))
                            buffer_updates["tool_calls"] = tool_call_entries
                            break

                    # Handle abort_after_execution (takes precedence over pause_after_execution)
                    if abort_after_tools:
                        tool_result, tool_func, tool_name, tool_id, arguments = abort_after_tools[0]
                        print(f"Aborting execution after tool: {tool_name}")

                        # Mark loop as complete with the tool result as final output
                        if hasattr(state, "final_output"):
                            state.final_output = tool_result.result
                        if hasattr(state, "is_complete"):
                            state.is_complete = True
                        is_complete = True
                        buffer_updates["is_complete"] = True
                        buffer_updates["final_output"] = state.final_output
                        buffer_updates["tool_calls"] = tool_call_entries

                        # Break out of the tool loop immediately
                        break

                    # Handle pause_after_execution
                    elif pause_after_tools:
                        tool_result, tool_func, tool_name, tool_id, arguments = pause_after_tools[0]
                        print(f"Pausing execution after tool: {tool_name}")
                        state.is_paused = True
                        state.paused_tool_id = tool_id
                        state.paused_tool_name = tool_name
                        state.paused_tool_arguments = arguments
                        state.paused_after_execution = True
                        state.paused_tool_result = tool_result

                        # Update chain status to PAUSE
                        self.graph._update_chain_status(ChainStatus.PAUSE)

                        # CRITICAL: Save pause state to checkpoint
                        self._capture_state_checkpoint(node_name, state)

                        # Return early with updated state
                        return {"state": state}

                    # Add to buffer updates
                    buffer_updates["tool_calls"] = tool_call_entries
                else:
                    # No tool use, this is the final response
                    print("Response does not contain tool calls, checking for final response")

                    # Check if we have an output schema and validate against it
                    output_schema = getattr(node, "output_schema", None)

                    final_content = content.strip() if content else None

                    if output_schema and final_content:
                        is_valid, validated_model, error_msg = self._validate_message_against_schema(
                            content, output_schema
                        )
                        if is_valid:
                            state.final_output = validated_model
                            final_content = validated_model.model_dump_json()
                        else:
                            logger.debug(
                                "LLM message does not validate against output schema, providing feedback to model."
                            )
                            assistant_message = LLMMessage(
                                role="assistant",
                                content=f"Error: Final message does not contain any tools "
                                f"calls or a valid output schema. Final message: {final_content}",
                                should_show_to_user=True,
                                id=getattr(response, "id", None)
                                if not isinstance(response, dict)
                                else response.get("id"),
                            )
                            # Add to messages
                            state.messages.append(assistant_message)

                            # Call the on_message callback if provided
                            self._on_message(node, assistant_message)
                            continue

                    # Create assistant message for final response
                    if final_content:  # Only add if content is not empty
                        assistant_message = LLMMessage(
                            role="assistant",
                            content=final_content,
                            should_show_to_user=True,
                            id=getattr(response, "id", None) if not isinstance(response, dict) else response.get("id"),
                        )

                        # Add to messages
                        state.messages.append(assistant_message)
                        if not state.final_output:
                            state.final_output = final_content

                        # Call the on_message callback if provided
                        self._on_message(node, assistant_message)

                        break

            except Exception as e:
                # Record error
                error = str(e)
                print(f"Error in tool loop: {error}")
                if hasattr(state, "error"):
                    state.error = error
                if hasattr(state, "is_complete"):
                    state.is_complete = True
                buffer_updates["error"] = error
                buffer_updates["is_complete"] = True

                # Add error message to state
                if hasattr(state, "messages") and error and error.strip():  # Only add if error is not empty
                    state.messages.append(
                        LLMMessage(role="system", content=f"Error: {error}", should_show_to_user=False)
                    )

                # Call the on_message callback for the error
                self._on_message(
                    node,
                    {
                        "message_type": "system",
                        "content": f"Error: {error}",
                        "is_final": True,
                        "is_error": True,
                        "iteration": current_iteration,
                        "timestamp": time.time(),
                    },
                )
                break

            # Increment iteration counter
            current_iteration += 1
            if hasattr(state, "current_iteration"):
                state.current_iteration = current_iteration
            buffer_updates["current_iteration"] = current_iteration

        # Check if we've reached max iterations
        if current_iteration >= max_iterations and not is_complete:
            print(f"Reached maximum iterations ({max_iterations})")
            max_iter_message = f"Reached maximum iterations ({max_iterations})"
            if hasattr(state, "is_complete"):
                state.is_complete = True
            if hasattr(state, "final_output"):
                state.final_output = max_iter_message
            buffer_updates["is_complete"] = True
            buffer_updates["final_output"] = max_iter_message

            # Call the on_message callback for the max iterations message
            self._on_message(
                node,
                {
                    "message_type": "system",
                    "content": max_iter_message,
                    "is_final": True,
                    "is_error": False,
                    "iteration": current_iteration,
                    "timestamp": time.time(),
                },
            )

            # Add a system message about max iterations
            if hasattr(state, "messages"):
                max_iter_message = f"Reached maximum iterations ({max_iterations})"
                if max_iter_message and max_iter_message.strip():  # Only add if message is not empty
                    state.messages.append(LLMMessage(role="system", content=max_iter_message, should_show_to_user=True))

        # Save final checkpoint
        self._capture_state_checkpoint(node_name, state)

        # Ensure buffer_updates has all needed fields
        buffer_updates["is_complete"] = True
        if "tool_calls" in buffer_updates:
            buffer_updates["tool_calls"] = tool_call_entries

        # Update state with buffer updates
        for key, value in buffer_updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

        # Print summary
        print("Tool node execution complete:")
        print(f"- Tool calls: {len(tool_call_entries)}")
        print(f"- Final messages: {len(state.messages)}")
        print(f"- Is complete: {is_complete}")
        if error:
            print(f"- Error: {error}")

        # Return buffer updates
        return buffer_updates

    def _capture_state_checkpoint(self, node_name: str, state: Any) -> None:
        """
        Helper method to capture the current state in a checkpoint.

        Args:
            node_name: Current node name
            state: Current state object
        """
        # Make sure this state is the current graph state
        self.graph.state = state

        # Ensure chain status is consistent with is_paused
        if hasattr(state, "is_paused") and state.is_paused and self.graph.chain_status != ChainStatus.PAUSE:
            print("[ToolEngine._capture_state_checkpoint] Setting chain status to PAUSE for paused state")
            self.graph._update_chain_status(ChainStatus.PAUSE)

        # Get the full state and save checkpoint
        try:
            engine_state = self.get_full_state()
            self.graph._save_checkpoint(node_name, engine_state)
        except Exception as e:
            print(f"[ToolEngine._capture_state_checkpoint] Error saving checkpoint: {str(e)}")
            # Continue execution despite checkpoint failure
            # This prevents serialization errors from breaking the entire workflow

    async def _execute_all(self) -> None:
        """
        Process all pending execution frames.
        Override to add debugging.
        """
        print(f"[ToolEngine._execute_all] Starting execution of {len(self.execution_frames)} frames")

        while self.execution_frames and self.graph.chain_status == ChainStatus.RUNNING:
            if len(self.execution_frames) > 1:
                print(f"[ToolEngine._execute_all] Executing {len(self.execution_frames)} frames in parallel")
                await asyncio.gather(
                    *(self._execute_frame(frame) for frame in self.execution_frames if frame is not None)
                )
            else:
                frame = self.execution_frames.pop(0)
                if frame is not None:
                    print(f"[ToolEngine._execute_all] Executing single frame: {frame.node_id}")
                    await self._execute_frame(frame)

            # Check if any frame has paused execution - if so, break the loop
            if hasattr(self.graph, "state") and hasattr(self.graph.state, "is_paused") and self.graph.state.is_paused:
                print("[ToolEngine._execute_all] Detected paused state, stopping execution")
                self.graph._update_chain_status(ChainStatus.PAUSE)
                break

        print(f"[ToolEngine._execute_all] Execution complete, frames remaining: {len(self.execution_frames)}")

    async def _execute_frame(self, frame: ExecutionFrame) -> None:
        """
        Execute a single frame to process a path through the graph.
        Override to add our own frame processing.
        """
        print(f"[ToolEngine._execute_frame] Executing frame: {frame.node_id}")

        # Special handling for tool nodes - we don't have to go through the parent method
        node_id = frame.node_id

        if node_id == START:
            # For START node, set up the next node and return
            print("[ToolEngine._execute_frame] Processing START node")
            children = self.graph.edges_map.get(node_id, [])
            if children:
                # Get the next node and create a new frame for it
                next_node_id = children[0]
                print(f"[ToolEngine._execute_frame] Setting up next node: {next_node_id}")
                frame.node_id = next_node_id
                frame.current_node = self.graph.nodes.get(next_node_id)
                # Keep this frame in the execution_frames list
                if frame not in self.execution_frames:
                    self.execution_frames.append(frame)
            else:
                print("[ToolEngine._execute_frame] START node has no children!")
        elif node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            frame.current_node = node

            # Handle tool node specially
            if hasattr(node, "is_tool_node") and node.is_tool_node:
                print(f"[ToolEngine._execute_frame] Node {node_id} is a tool node, executing")

                # Execute the tool node
                await self.execute_node(frame)

                # Get next node
                children = self.graph.edges_map.get(node_id, [])
                if children:
                    next_node_id = children[0]
                    frame.node_id = next_node_id
                    frame.current_node = self.graph.nodes.get(next_node_id)
                    print(f"[ToolEngine._execute_frame] Moving to next node: {next_node_id}")

                    # If we're moving to END, continue execution
                    if next_node_id == END:
                        # For other nodes, use parent method for END handling
                        frame.node_id = END
                        frame.current_node = None
                        await super()._execute_frame(frame)
                else:
                    print(f"[ToolEngine._execute_frame] Node {node_id} has no children!")
            else:
                # For other nodes, use parent method
                await super()._execute_frame(frame)
        else:
            # For other nodes, use parent method
            await super()._execute_frame(frame)

        print(f"[ToolEngine._execute_frame] Frame execution complete: {frame.node_id}")

    async def execute_node(self, frame: ExecutionFrame) -> Dict[str, Any]:
        """
        Execute a node in the graph.

        Overrides the base execute_node method to handle tool node execution.

        Args:
            frame: Current execution frame

        Returns:
            Dictionary of buffer updates
        """
        node = frame.current_node

        # If node is not set, try to get it from the graph using node_id
        if node is None and frame.node_id in self.graph.nodes:
            node = self.graph.nodes[frame.node_id]
            frame.current_node = node

        if node is None:
            raise ValueError(f"Cannot execute node: Node not found for ID {frame.node_id}")

        print(f"[ToolEngine.execute_node] Executing node: {node.name}, type: {type(node)}")

        # If this is a ToolNode, handle it specially
        if isinstance(node, ToolNode):
            print(f"[ToolEngine.execute_node] Node {node.name} is a ToolNode, delegating to _execute_tool_node")
            return await self._execute_tool_node(frame)

        # Otherwise, use the standard node execution
        print(f"[ToolEngine.execute_node] Node {node.name} is a standard node, delegating to parent")
        return await super().execute_node(frame)
