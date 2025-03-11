"""
Module for LLM tool nodes that support function calling patterns.

This module provides a specialized node type for LLM interactions with tools/function calling,
operating in a loop pattern. It integrates with the primeGraph system while providing
a separate execution path specifically designed for LLM tool interactions.
"""

import asyncio
import time
import inspect
import json
from enum import Enum
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, get_type_hints, Type

from pydantic import BaseModel, Field, create_model

from primeGraph.graph.base import Node
from primeGraph.graph.executable import Graph
from primeGraph.graph.engine import Engine, ExecutionFrame
from primeGraph.graph.llm_clients import LLMClientBase, Provider, LLMClientFactory
from primeGraph.models.state import GraphState
from primeGraph.models.checkpoint import ChainStatus
from primeGraph.checkpoint.base import CheckpointData, StorageBackend
from primeGraph.buffer.base import BaseBuffer
from primeGraph.buffer.history import HistoryBuffer
from primeGraph.buffer.last_value import LastValueBuffer
from primeGraph.buffer.factory import History, LastValue
from primeGraph.constants import START, END


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
    func: Optional[Callable] = None

    model_config = {
        "arbitrary_types_allowed": True
    }


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


class LLMMessage(BaseModel):
    """Message in an LLM conversation"""
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    model_config = {
        "extra": "allow"  # Allow additional fields not specified in the model
    }


class ToolLoopOptions(BaseModel):
    """Options for configuring a tool loop execution"""
    max_iterations: int = 10
    timeout_seconds: Optional[float] = None
    max_tokens: int = 4096
    stop_on_first_error: bool = False
    trace_enabled: bool = False


class ToolState(GraphState):
    """Base state for tool loops, storing messages, tool calls, and results"""
    messages: History[LLMMessage] = Field(default_factory=list)
    tool_calls: History[ToolCallLog] = Field(default_factory=list)
    current_iteration: LastValue[int] = 0
    max_iterations: LastValue[int] = 10
    is_complete: LastValue[bool] = False
    final_output: LastValue[Optional[str]] = None
    error: LastValue[Optional[str]] = None
    current_trace: LastValue[Optional[Dict[str, Any]]] = None
    last_token_usage: LastValue[Optional[Dict[str, int]]] = None


def tool(description: str, tool_type: ToolType = ToolType.FUNCTION):
    """
    Decorator to mark a function as a tool available to LLMs.
    
    This decorator extracts type hints and parameter information from the function
    to generate a schema that LLMs can use to call the function appropriately.
    The decorated function can then be passed to a ToolNode for execution in
    response to LLM requests.
    
    Args:
        description: Description of what the tool does (shown to the LLM)
        tool_type: Type of tool (function, action, retrieval)
    
    Examples:
        ```python
        @tool("Get weather for a location")
        async def get_weather(location: str, unit: str = "celsius") -> Dict:
            # Implementation
            return {"temperature": 22.5, "conditions": "sunny"}
        ```
    """
    def decorator(func):
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Create parameter schema based on type hints
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param.name == "self" or param.name == "cls":
                continue

            param_type = hints.get(param_name, Any)
            is_required = param.default == inspect.Parameter.empty

            if is_required:
                required.append(param_name)

            # Handle type conversions for parameters
            if param_type in (str, int, float, bool):
                properties[param_name] = (param_type, ... if is_required else None)
            elif isinstance(param_type, type) and issubclass(param_type, BaseModel):
                properties[param_name] = (param_type, ... if is_required else None)
            else:
                properties[param_name] = (Any, ... if is_required else None)

        # Create schema for the tool
        schema = {
            "name": func.__name__,
            "description": description,
            "type": tool_type,
            "parameters": properties,
            "required_params": required,
            "func": func,
        }

        # Store schema on the function
        func._tool_definition = ToolDefinition(**schema)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # The actual tool execution is handled separately
            return await func(*args, **kwargs)

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
            router_paths=None
        )
        
        # Add tool node specific attributes
        instance.tools = tools
        instance.llm_client = llm_client
        instance.options = options or ToolLoopOptions()
        instance.state_class = state_class
        instance.on_tool_use = on_tool_use
        instance.is_tool_node = True
        
        # Validate tools
        for i, tool_func in enumerate(instance.tools):
            if not hasattr(tool_func, "_tool_definition"):
                raise ValueError(
                    f"Tool at index {i} ({tool_func.__name__}) is not decorated with @tool"
                )
                
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
            
            # Create parameter schema
            param_model = create_model(
                f"{tool_def.name}Params", 
                **{k: (v[0], v[1]) for k, v in tool_def.parameters.items()}
            )
            json_schema = param_model.model_json_schema()
            
            # Format for provider
            if provider and provider.lower() == "anthropic":
                schema = {
                    "type": "custom",
                    "custom": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "input_schema": json_schema
                    }
                }
            elif provider and provider.lower() == "google":
                schema = {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": json_schema
                }
            else:
                # Default to OpenAI format
                schema = {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": json_schema
                    }
                }
            
            schemas.append(schema)
        
        return schemas
    
    def find_tool_by_name(self, name: str) -> Optional[Callable]:
        """Find a tool by name from this node's tool list"""
        for tool_func in self.tools:
            if tool_func._tool_definition.name == name:
                return tool_func
        return None
    
    async def execute_tool(
        self, 
        tool_func: Callable, 
        arguments: Dict[str, Any], 
        tool_id: str
    ) -> ToolCallLog:
        """
        Execute a tool function and record the execution.
        
        Args:
            tool_func: The tool function to execute
            arguments: Arguments to pass to the tool
            tool_id: Unique ID for this tool call
            
        Returns:
            ToolCallLog with the execution results
        """
        tool_name = tool_func._tool_definition.name
        start_time = time.time()
        success = True
        error = None
        result = None
        
        try:
            # Execute the tool
            result = await tool_func(**arguments)
        except Exception as e:
            success = False
            error = str(e)
            result = f"Error: {error}"
        
        # Calculate duration
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Create log entry
        log = ToolCallLog(
            id=tool_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            timestamp=start_time,
            duration_ms=duration_ms,
            error=error
        )
        
        # Call the callback if provided
        if self.on_tool_use:
            self.on_tool_use({
                "id": tool_id,
                "name": tool_name,
                "arguments": arguments,
                "success": success,
                "result": result,
                "error": error
            })
        
        return log


class ToolGraph(Graph):
    """
    Specialized graph for tool-based LLM workflows.
    
    This extends the base Graph with functionality specific to tool-based
    execution, making it easier to create and work with tool-based workflows.
    """
    
    def __init__(
        self,
        name: str,
        state_class: Type[GraphState] = ToolState,
        max_iterations: int = 10,
        checkpoint_storage: Optional[StorageBackend] = None,
        execution_timeout: int = 60 * 5,
        max_node_iterations: int = 100,
        verbose: bool = False,
    ):
        """
        Initialize a tool-specific graph.
        
        Args:
            name: Name of the graph
            state_class: State class for the graph
            max_iterations: Maximum iterations for tool loops
            checkpoint_storage: Optional storage backend for checkpoints
            execution_timeout: Timeout for execution in seconds
            max_node_iterations: Maximum iterations per node
            verbose: Whether to enable verbose logging
        """
        # Initialize state from the state class
        state = state_class()
        
        # Set graph name and state class as properties
        self.name = name
        self.state_class = state_class
        
        # Call parent constructor with all required parameters
        super().__init__(
            state=state,
            checkpoint_storage=checkpoint_storage,
            execution_timeout=execution_timeout,
            max_node_iterations=max_node_iterations,
            verbose=verbose
        )
        
        self.max_iterations = max_iterations
        
        # START and END nodes are added in the BaseGraph constructor
        
    def add_tool_node(
        self,
        name: str,
        tools: List[Callable],
        llm_client: Any,
        options: Optional[ToolLoopOptions] = None,
        on_tool_use: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ToolNode:
        """
        Add a tool node to the graph.
        
        Args:
            name: Name of the node
            tools: List of tool functions
            llm_client: LLM client instance for provider API calls
            options: Tool loop options
            on_tool_use: Optional callback for tool use
            
        Returns:
            The created ToolNode
        """
        if options is None:
            options = ToolLoopOptions(max_iterations=self.max_iterations)
            
        node = ToolNode(
            name=name,
            tools=tools,
            llm_client=llm_client,
            options=options,
            state_class=self.state_class,
            on_tool_use=on_tool_use,
        )
        
        # Add the node directly to the nodes dictionary
        self.nodes[name] = node
        return node
        
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
            node = self.add_tool_node(
                name=name, 
                tools=node_tools,
                llm_client=llm_client,
                options=node_options
            )
            nodes.append(node)
            
        # Connect nodes linearly
        for i in range(len(nodes) - 1):
            self.add_edge(nodes[i].name, nodes[i+1].name)
            
        # Connect to START and END if requested
        if connect_to_start_end:
            if nodes:
                self.add_edge(START, nodes[0].name)
                self.add_edge(nodes[-1].name, END)
                
        return nodes


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
    
    This engine should be used for any graph containing ToolNode instances.
    """
    
    async def execute(self, initial_state: Optional[GraphState] = None) -> Any:
        """
        Begin executing the graph from the START node with optional initial state.
        
        Args:
            initial_state: Optional initial state to use
            
        Returns:
            Result of execution
        """
        print(f"\n[ToolEngine.execute] Starting execution with initial_state: {initial_state}")
        
        self._has_executed = True  # Mark that execute() has been run
        
        # Use provided initial state if given, otherwise use graph's state
        state_to_use = initial_state if initial_state is not None else self.graph.state
        print(f"[ToolEngine.execute] Using state: {type(state_to_use)}")
        
        # Log graph nodes for debugging
        print(f"[ToolEngine.execute] Graph nodes: {list(self.graph.nodes.keys())}")
        
        # Log edge mapping for debugging
        print(f"[ToolEngine.execute] Graph edges: {self.graph.edges_map}")
        
        # Create the initial frame with the START node
        initial_frame = ExecutionFrame(START, state_to_use)
        self.execution_frames.append(initial_frame)
        print(f"[ToolEngine.execute] Created initial frame with START node")
        
        # Update graph status to running
        self.graph._update_chain_status(ChainStatus.RUNNING)
        
        # Execute all frames
        print(f"[ToolEngine.execute] Calling _execute_all")
        await self._execute_all()
        print(f"[ToolEngine.execute] _execute_all completed")
        
        # Return a result object with the final state
        print(f"[ToolEngine.execute] Returning result with state: {state_to_use}")
        return type('ExecutionResult', (), {'state': state_to_use, 'chain_id': self.graph.chain_id})

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
                result = await self.execute_node(frame)
                
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
    
    async def _execute_tool_node(self, frame: ExecutionFrame) -> Dict[str, Any]:
        """
        Execute a tool node with its LLM interaction loop.
        
        Args:
            frame: Current execution frame
            
        Returns:
            Dictionary of buffer updates
        """
        node = frame.current_node
        
        # If node is not set, try to get it from the graph
        if node is None and frame.node_id in self.graph.nodes:
            node = self.graph.nodes[frame.node_id]
            frame.current_node = node
            
        # Make sure node is a ToolNode
        if not isinstance(node, ToolNode):
            raise TypeError(f"Expected ToolNode, got {type(node)}")
            
        # Make sure we have a state to work with
        if frame.state is None:
            raise ValueError("Frame state is None, cannot execute tool node")
            
        # For clarity, get the state
        state = frame.state
        
        # Initialize variables
        current_iteration = 0
        max_iterations = node.options.max_iterations
        is_complete = False
        error = None
        buffer_updates = {}
        
        # Print debug info
        print(f"\nExecuting tool node: {node.name}")
        print(f"State type: {type(state)}")
        print(f"State fields: {state.model_fields.keys()}")
        
        # Check if the node has the LLM client set
        if not hasattr(node, 'llm_client') or node.llm_client is None:
            error = "LLM client not set for tool node"
            buffer_updates["error"] = error
            buffer_updates["is_complete"] = True
            return buffer_updates
        
        # Get messages from state
        try:
            messages_list = getattr(state, "messages", [])
            if hasattr(messages_list, "get") and callable(messages_list.get):
                messages_list = messages_list.get()
            messages = [msg.model_dump() for msg in messages_list]
            print(f"Found {len(messages)} messages in state")
        except Exception as e:
            print(f"Error getting messages: {e}")
            messages = []
            
        if not messages:
            # No messages to work with
            error = "No messages found in state"
            buffer_updates["error"] = error
            buffer_updates["is_complete"] = True
            return buffer_updates
            
        # Get provider-specific tool schemas
        provider = "openai"  # Default
        if hasattr(node.llm_client, "client"):
            # Try to determine provider from client type
            client_module = node.llm_client.client.__class__.__module__
            if "anthropic" in client_module:
                provider = "anthropic"
            elif "openai" in client_module:
                provider = "openai"
        
        print(f"Using provider: {provider}")
        
        # Get tool schemas for the provider
        tool_schemas = node.get_tool_schemas(provider)
        print(f"Generated {len(tool_schemas)} tool schemas")
        
        # Record in state that we've started
        buffer_updates["current_iteration"] = current_iteration
        buffer_updates["max_iterations"] = max_iterations
        buffer_updates["is_complete"] = is_complete
        
        # Set timeout if specified
        timeout_s = node.options.timeout_seconds
        start_time = time.time()
        
        # Create a list to collect tool call entries
        tool_call_entries = []
        
        # Main LLM interaction loop
        while current_iteration < max_iterations and not is_complete and not error:
            print(f"\nTool loop iteration {current_iteration + 1}/{max_iterations}")
            
            # Check timeout if specified
            if timeout_s and (time.time() - start_time) > timeout_s:
                buffer_updates["error"] = f"Execution timed out after {timeout_s}s"
                break
                
            # Save checkpoint after each iteration
            if hasattr(self.graph, "_save_checkpoint"):
                self.graph._save_checkpoint(frame.node_id, self.get_full_state())
            
            try:
                # Extract model options from node.options
                model_kwargs = {}
                if hasattr(node.options, "model_dump"):
                    try:
                        # Only include parameters that are valid for LLM API calls 
                        model_kwargs = {"max_tokens": node.options.max_tokens}
                    except Exception as e:
                        print(f"Error extracting model options: {e}")
                
                # Generate response from LLM
                print(f"Calling LLM generate with {len(messages)} messages and {len(tool_schemas)} tools")
                content, response = await node.llm_client.generate(
                    messages=messages,
                    tools=tool_schemas,
                    **model_kwargs
                )
                
                # Check if response requires tool use
                if node.llm_client.is_tool_use_response(response):
                    print("Response contains tool calls")
                    
                    # Get tool calls
                    tool_calls = node.llm_client.extract_tool_calls(response)
                    
                    # Prepare assistant message with tool calls
                    if provider == "openai":
                        # For OpenAI, we need to include the tool_calls in the assistant message
                        assistant_message = {
                            "role": "assistant",
                            "content": content or "",
                            "tool_calls": [{
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call["arguments"])
                                }
                            } for call in tool_calls]
                        }
                    else:
                        # For other providers
                        assistant_message = {
                            "role": "assistant",
                            "content": content or "",
                        }
                    
                    # Append to messages list for next iteration
                    messages.append(assistant_message)
                    
                    # Add to buffer updates for state update
                    if "messages" not in buffer_updates:
                        buffer_updates["messages"] = list(messages_list)
                    buffer_updates["messages"].append(LLMMessage(**assistant_message))
                    
                    # Process each tool call
                    for tool_call in tool_calls:
                        tool_id = tool_call["id"]
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]
                        
                        print(f"Processing tool call: {tool_name}({tool_args})")
                        
                        # Find the tool function
                        tool_func = node.find_tool_by_name(tool_name)
                        
                        if tool_func:
                            # Execute the tool
                            print(f"Executing tool: {tool_name}")
                            tool_result = await node.execute_tool(tool_func, tool_args, tool_id)
                            
                            # Add to tool call entries
                            tool_call_entries.append(tool_result)
                            
                            # Add tool response to messages - format depends on provider
                            if provider == "anthropic":
                                # For Anthropic, use a simpler format
                                tool_message = {
                                    "role": "user",
                                    "content": f"Tool result for {tool_name}: {str(tool_result.result)}"
                                }
                            else:
                                # For OpenAI, use the standard format
                                tool_message = {
                                    "role": "tool",
                                    "content": str(tool_result.result),
                                    "tool_call_id": tool_id
                                }
                                
                            # Append tool message to messages for next iteration
                            messages.append(tool_message)
                            
                            # Add to buffer updates for state update
                            buffer_updates["messages"].append(LLMMessage(**tool_message))
                        else:
                            # Tool not found
                            error_msg = f"Tool '{tool_name}' not found"
                            print(f"Error: {error_msg}")
                            
                            # Create error tool call log
                            tool_result = ToolCallLog(
                                id=tool_id,
                                tool_name=tool_name,
                                arguments=tool_args,
                                result=f"Error: {error_msg}",
                                success=False,
                                timestamp=time.time(),
                                error=error_msg
                            )
                            
                            # Add to tool call entries
                            tool_call_entries.append(tool_result)
                            
                            # Add error response to messages
                            tool_message = {
                                "role": "tool" if provider != "anthropic" else "user",
                                "content": f"Error: {error_msg}"
                            }
                            
                            if provider != "anthropic":
                                tool_message["tool_call_id"] = tool_id
                                
                            # Append tool message to messages
                            messages.append(tool_message)
                            
                            # Add to buffer updates for state update
                            buffer_updates["messages"].append(LLMMessage(**tool_message))
                    
                    # Update tool_calls in buffer_updates
                    buffer_updates["tool_calls"] = tool_call_entries
                    
                    # Continue to next iteration
                    current_iteration += 1
                    buffer_updates["current_iteration"] = current_iteration
                    
                else:
                    # No tool use, this is the final response
                    print("Response does not contain tool calls, finishing")
                    
                    # Create assistant message for final response
                    assistant_message = {"role": "assistant", "content": content}
                    
                    # Add to messages
                    messages.append(assistant_message)
                    
                    # Add to buffer updates
                    if "messages" not in buffer_updates:
                        buffer_updates["messages"] = list(messages_list)
                    buffer_updates["messages"].append(LLMMessage(**assistant_message))
                    
                    # Update tool_calls if any were made
                    if tool_call_entries:
                        buffer_updates["tool_calls"] = tool_call_entries
                    
                    # Mark as complete with final output
                    buffer_updates["final_output"] = content
                    buffer_updates["is_complete"] = True
                    is_complete = True
                    
                    # Save the raw response for the user to access
                    buffer_updates["current_trace"] = {
                        "raw_response": response
                    }
                    
            except Exception as e:
                # Record error
                error = str(e)
                print(f"Error in tool loop: {error}")
                buffer_updates["error"] = error
                buffer_updates["is_complete"] = True
                is_complete = True
                
            # Check if we've reached max iterations
            if current_iteration >= max_iterations and not is_complete:
                print(f"Reached maximum iterations ({max_iterations})")
                buffer_updates["is_complete"] = True
                buffer_updates["final_output"] = f"Reached maximum iterations ({max_iterations})"
                is_complete = True
                
        # Save final checkpoint
        if hasattr(self.graph, "_save_checkpoint"):
            self.graph._save_checkpoint(frame.node_id, self.get_full_state())
        
        # Ensure is_complete is set
        buffer_updates["is_complete"] = True
        
        # Update our frame.state directly (needed because we're using custom state objects)
        for key, value in buffer_updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        # Print summary
        print(f"Tool node execution complete:")
        print(f"- Tool calls: {len(tool_call_entries)}")
        print(f"- Final messages: {len(messages)}")
        print(f"- Is complete: {is_complete}")
        if error:
            print(f"- Error: {error}")
        
        # Return buffer updates for the engine
        return buffer_updates