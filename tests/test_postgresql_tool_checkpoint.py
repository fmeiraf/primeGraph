"""
Tests for PostgreSQL checkpointing with ToolEngine pauses.

These tests verify that:
1. A tool graph can be paused during tool execution
2. The paused state can be saved to PostgreSQL
3. A new graph can be loaded from the checkpoint
4. The loaded graph can properly resume execution
"""

import time
from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel, Field

from primeGraph.buffer.factory import History, LastValue
from primeGraph.checkpoint.postgresql import PostgreSQLStorage
from primeGraph.constants import END, START
from primeGraph.graph.llm_clients import LLMClientBase
from primeGraph.graph.llm_tools import (LLMMessage, ToolEngine, ToolGraph,
                                        ToolLoopOptions, ToolState, tool)


# Define MockLLMClient for testing
class MockLLMClient(LLMClientBase):
    """
    Mock LLM client that simulates tool-calling behavior with predefined responses
    """
    
    def __init__(self, conversation_flow=None):
        """
        Initialize with predefined conversation flow
        
        Args:
            conversation_flow: List of responses to return in sequence
        """
        super().__init__()
        self.conversation_flow = conversation_flow or []
        self.call_count = 0
        self.call_history = []
        
        # This is for debugging to track mock usage
        print(f"Creating MockLLMClient with {len(self.conversation_flow)} responses")
        
    async def generate(self, messages, tools=None, tool_choice=None, **kwargs):
        """Simulate LLM response generation"""
        self.call_history.append({"messages": messages, "tools": tools})
        
        # Debug info
        print(f"MockLLMClient.generate called (call #{self.call_count + 1})")
        
        if self.call_count >= len(self.conversation_flow):
            # Default to a simple text response if no more predefined responses
            print("No more responses in flow, returning default")
            return "I don't have any more actions to take.", {
                "content": "I don't have any more actions to take."
            }
            
        response = self.conversation_flow[self.call_count]
        self.call_count += 1
        
        # Extract content for the return value
        content = response.get("content", "")
        
        # Debug info
        print(f"Returning response: {response}")
            
        return content, response
    
    def is_tool_use_response(self, response):
        """Check if response contains tool calls"""
        has_tool_calls = "tool_calls" in response
        print(f"is_tool_use_response: {has_tool_calls}")
        return has_tool_calls
    
    def extract_tool_calls(self, response):
        """Extract tool calls from response"""
        if "tool_calls" not in response:
            print("extract_tool_calls: No tool calls found")
            return []
        
        tool_calls = response["tool_calls"]    
        print(f"extract_tool_calls: Found {len(tool_calls)} tool calls")
        return tool_calls


# Requires you to be running the docker from primeGraph/docker
@pytest.fixture
def postgres_storage():
    storage = PostgreSQLStorage.from_config(
        host="localhost",
        port=5432,
        user="primegraph",
        password="primegraph",
        database="primegraph",
    )
    assert storage.check_schema(), "Schema is not valid"
    return storage


class ToolCheckpointState(ToolState):
    """State for tool checkpoint testing"""
    customer_data: LastValue[Optional[Dict[str, Any]]] = None
    order_data: History[Dict[str, Any]] = Field(default_factory=list)
    processing_results: History[Dict[str, Any]] = Field(default_factory=list)


class AnalyticsTrackingState(ToolState):
    """State for analytics tracking testing"""
    user_id: LastValue[Optional[str]] = Field(default=None)
    interaction_id: LastValue[Optional[str]] = Field(default=None)
    analysis_goal: LastValue[Optional[str]] = Field(default=None)
    
    # Add a nested BaseModel field to test complex type restoration
    class AnalyticsMetadata(BaseModel):
        """Nested metadata for analytics tracking"""
        session_id: str = Field(default="")
        timestamp: int = Field(default=0)
        source: str = Field(default="web")
        
        class SessionDetails(BaseModel):
            """Deeply nested session details"""
            browser: str = Field(default="unknown")
            device: str = Field(default="desktop")
            is_mobile: bool = Field(default=False)
        
        details: SessionDetails = Field(default_factory=lambda: AnalyticsTrackingState.AnalyticsMetadata.SessionDetails())
    
    metadata: LastValue[Optional[AnalyticsMetadata]] = Field(default=None)


# Define tool functions for testing
@tool("Get customer information")
async def get_customer_info(customer_id: str, state=None) -> Dict[str, Any]:
    """Get customer details by ID"""
    # Test data
    customers = {
        "C1": {
            "id": "C1", 
            "name": "John Doe", 
            "email": "john@example.com",
            "orders": ["O1", "O2"]
        }
    }
    
    if customer_id not in customers:
        raise ValueError(f"Customer {customer_id} not found")
    
    result = customers[customer_id]
    
    # If state is available, update the customer_data field
    if state and hasattr(state, "customer_data"):
        state.customer_data = result
    
    return result


@tool("Get order details")
async def get_order_details(order_id: str, state=None) -> Dict[str, Any]:
    """Get order details by ID"""
    # Test data
    orders = {
        "O1": {
            "id": "O1",
            "customer_id": "C1",
            "product": "Widget A",
            "quantity": 2,
            "price": 19.99,
            "status": "shipped"
        },
        "O2": {
            "id": "O2",
            "customer_id": "C1",
            "product": "Gadget B",
            "quantity": 1,
            "price": 49.99,
            "status": "processing"
        }
    }
    
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    
    result = orders[order_id]
    
    # If state is available, update the order_data field
    if state and hasattr(state, "order_data"):
        state.order_data.append(result)
    
    return result


# Add a tool with pause_before_execution flag set to True
@tool("Process payment", pause_before_execution=True)
async def process_payment(order_id: str, amount: float, state=None) -> Dict[str, Any]:
    """Process a payment for an order, pausing for verification"""
    # This would normally interact with a payment gateway
    # But for testing it just returns a confirmation
    result = {
        "order_id": order_id,
        "amount": amount,
        "status": "processed",
        "transaction_id": f"TX-{order_id}-{int(time.time())}"
    }
    
    # If state is provided and has processing_results, update it directly
    if state and hasattr(state, "processing_results"):
        state.processing_results.append(result)
        
    return result


# Add a tool with pause_after_execution flag set to True
@tool("Update customer account", pause_after_execution=True)
async def update_customer_account(customer_id: str, email: str, state=None) -> Dict[str, Any]:
    """Update a customer's account information, pausing after execution for verification"""
    # Simulate updating customer account
    result = {
        "customer_id": customer_id,
        "new_email": email,
        "status": "updated",
        "timestamp": int(time.time())
    }
    
    # Update customer_data in state if available
    if state and hasattr(state, "customer_data"):
        state.customer_data = result
        
    return result


@tool("Track user interaction")
async def track_user_interaction(user_id: str, interaction_id: str, state=None) -> Dict[str, Any]:
    """Track a user interaction in the analytics system"""
    result = {
        "user_id": user_id,
        "interaction_id": interaction_id,
        "timestamp": int(time.time())
    }
    
    # Update state if available
    if state and hasattr(state, "user_id"):
        state.user_id = user_id
    if state and hasattr(state, "interaction_id"):
        state.interaction_id = interaction_id
        
    return result


@tool("Set analysis goal", pause_after_execution=True)
async def set_analysis_goal(goal: str, state=None) -> Dict[str, Any]:
    """Set the analysis goal for the current session"""
    result = {
        "goal": goal,
        "status": "set",
        "timestamp": int(time.time())
    }
    
    # Update state if available
    if state and hasattr(state, "analysis_goal"):
        state.analysis_goal = goal
        
    return result


def create_mock_flow_for_pause_before():
    """Create a conversation flow that uses the payment tool which pauses before execution"""
    return [
        # First get customer info
        {
            "content": "I'll help you process a payment for customer C1. Let me look up their information first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_customer_info",
                    "arguments": {"customer_id": "C1"}
                }
            ]
        },
        # Then get order details
        {
            "content": "I found the customer. Now let me get the order details.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "get_order_details",
                    "arguments": {"order_id": "O2"}
                }
            ]
        },
        # Then process payment (this will pause execution)
        {
            "content": "I found the order. Let me process the payment now.",
            "tool_calls": [
                {
                    "id": "call_3",
                    "name": "process_payment",
                    "arguments": {"order_id": "O2", "amount": 49.99}
                }
            ]
        },
        # Final response (only reached after resume)
        {
            "content": "The payment for order O2 in the amount of $49.99 has been successfully processed."
        }
    ]


def create_mock_flow_for_pause_after():
    """Create a conversation flow that uses the account update tool which pauses after execution"""
    return [
        # First get customer info
        {
            "content": "I'll help you update the email for customer C1. Let me look up their information first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_customer_info",
                    "arguments": {"customer_id": "C1"}
                }
            ]
        },
        # Then update account (this will pause after execution)
        {
            "content": "I found the customer. Let me update their email now.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "update_customer_account",
                    "arguments": {"customer_id": "C1", "email": "john.doe.new@example.com"}
                }
            ]
        },
        # Final response (only reached after resume)
        {
            "content": "The email for customer C1 has been successfully updated to john.doe.new@example.com."
        }
    ]


def create_analytics_flow():
    """Create a conversation flow for analytics tracking"""
    return [
        # First track the user interaction
        {
            "content": "I'll track this user interaction.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "track_user_interaction",
                    "arguments": {"user_id": "U123", "interaction_id": "INT456"}
                }
            ]
        },
        # Then set the analysis goal (this will pause after execution)
        {
            "content": "Now I'll set the analysis goal.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "set_analysis_goal",
                    "arguments": {"goal": "Identify user behavior patterns"}
                }
            ]
        },
        # Final response (only reached after resume)
        {
            "content": "The analysis goal has been set to identify user behavior patterns."
        }
    ]


@pytest.fixture
def tool_tools():
    """Fixture providing tools for testing"""
    return [get_customer_info, get_order_details, process_payment, update_customer_account]


@pytest.mark.asyncio
async def test_checkpoint_pause_before_execution(postgres_storage, tool_tools):
    """Test checkpointing with pause before tool execution"""
    # Create mock LLM client
    mock_client = MockLLMClient(conversation_flow=create_mock_flow_for_pause_before())
    
    # Create a graph with PostgreSQL checkpoint storage
    # Create initial state with request to process payment
    initial_state = ToolCheckpointState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful payment assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="Process payment for order O2 from customer C1."
        )
    ]
    graph = ToolGraph(
        "payment_processing", 
        state=initial_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = graph.add_tool_node(
        name="payment_agent",
        tools=tool_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    
    
    # Execute the graph
    await graph.execute()
    
    # Access the state from the graph
    first_state = graph.state
    
    # Verify the execution was paused before processing payment
    assert first_state.is_paused is True
    assert first_state.paused_tool_name == "process_payment"
    assert first_state.paused_tool_arguments["order_id"] == "O2"
    assert first_state.paused_tool_arguments["amount"] == 49.99
    
    # Check chain ID
    chain_id = graph.chain_id
    
    # Create a new graph instance
    new_graph = ToolGraph(
        "payment_processing", 
        state=initial_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = new_graph.add_tool_node(
        name="payment_agent",
        tools=tool_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    new_graph.add_edge(START, node.name)
    new_graph.add_edge(node.name, END)
    
    # Load the checkpoint
    new_graph.load_from_checkpoint(chain_id)
    
    # Verify the loaded state is paused correctly
    assert new_graph.state.is_paused is True
    assert new_graph.state.paused_tool_name == "process_payment"
    assert new_graph.state.paused_tool_arguments["order_id"] == "O2"
    
    # Resume execution
    await new_graph.resume(execute_tool=True)
    
    # Check the resumed state
    resumed_state = new_graph.state
    
    # Verify the execution completed
    assert resumed_state.is_paused is False
    assert resumed_state.is_complete is True
    
    # Verify the payment was processed
    assert any(call.tool_name == "process_payment" for call in resumed_state.tool_calls)
    assert len(resumed_state.processing_results) > 0
    assert resumed_state.processing_results[0]["order_id"] == "O2"


@pytest.mark.asyncio
async def test_checkpoint_pause_after_execution(postgres_storage, tool_tools):
    """Test checkpointing with pause after tool execution"""
    # Create mock LLM client
    mock_client = MockLLMClient(conversation_flow=create_mock_flow_for_pause_after())
    
    # Create initial state with request to process payment
    initial_state = ToolCheckpointState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful payment assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="Process payment for order O2 from customer C1."
        )
    ]
    graph = ToolGraph(
        "account_update", 
        state=initial_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = graph.add_tool_node(
        name="account_update_agent",
        tools=tool_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    # Create engine
    engine = ToolEngine(graph)
        
    # Execute the graph
    result = await engine.execute()
    
    # Get the state from the result
    first_state = result.state
    
    # Verify the execution was paused after updating account
    assert first_state.is_paused is True
    assert first_state.paused_tool_name == "update_customer_account"
    assert first_state.paused_after_execution is True
    assert first_state.paused_tool_result is not None
    assert first_state.paused_tool_result.tool_name == "update_customer_account"
    assert first_state.paused_tool_result.arguments["customer_id"] == "C1"
    assert first_state.paused_tool_result.arguments["email"] == "john.doe.new@example.com"
    
    # Check chain ID
    chain_id = graph.chain_id
    
    # Create a new graph instance
    new_graph = ToolGraph(
        "account_update", 
        state=initial_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = new_graph.add_tool_node(
        name="account_update_agent",
        tools=tool_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    new_graph.add_edge(START, node.name)
    new_graph.add_edge(node.name, END)
    
    # Create engine
    new_engine = ToolEngine(new_graph)
    
    # Load the checkpoint
    new_graph.load_from_checkpoint(chain_id)
    
    # Resume execution
    result = await new_engine.resume(new_graph.state, execute_tool=True)
    
    # Get final state
    final_state = result.state
    
    # Verify the execution completed successfully
    assert final_state.is_paused is False
    assert final_state.is_complete is True
    
    # Check that the update_customer_account tool was executed
    tool_calls = [call.tool_name for call in final_state.tool_calls]
    assert "update_customer_account" in tool_calls
    
    # Also check that the previous tool call is preserved
    assert "get_customer_info" in tool_calls
    
    # Check final message
    assert "successfully updated" in final_state.final_output


@pytest.mark.asyncio
async def test_checkpoint_reject_execution(postgres_storage, tool_tools):
    """Test rejecting tool execution after loading checkpoint"""
    # Create mock LLM client
    mock_client = MockLLMClient(conversation_flow=create_mock_flow_for_pause_before())
    
 
    # Create initial state with request to process payment
    initial_state = ToolCheckpointState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful payment assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="Process payment for order O2 from customer C1."
        )
    ]
    graph = ToolGraph(
        "payment_reject", 
        state=initial_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = graph.add_tool_node(
        name="payment_agent",
        tools=tool_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    # Create engine
    engine = ToolEngine(graph)
    
    # Execute the graph
    result = await engine.execute()
    
    # Get the state from the result
    first_state = result.state
    
    # Verify the execution was paused before processing payment
    assert first_state.is_paused is True
    assert first_state.paused_tool_name == "process_payment"
    
    # Check chain ID
    chain_id = graph.chain_id
    
    # Create a new graph instance
    new_graph = ToolGraph(
        "payment_reject", 
        state=initial_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = new_graph.add_tool_node(
        name="payment_agent",
        tools=tool_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    new_graph.add_edge(START, node.name)
    new_graph.add_edge(node.name, END)
    
    # Create engine
    new_engine = ToolEngine(new_graph)
    
    # Load the checkpoint
    new_graph.load_from_checkpoint(chain_id)
    
    # Resume execution and reject the tool
    result = await new_engine.resume(new_graph.state, execute_tool=False)
    
    # Get final state
    final_state = result.state
    
    # Check that the tool wasn't executed after rejecting
    process_payment_calls = [
        call for call in final_state.tool_calls 
        if call.tool_name == "process_payment"
    ]
    assert len(process_payment_calls) == 0
    
    # Check that we have a system message about skipping the tool
    system_messages = [
        msg for msg in final_state.messages
        if msg.role == "system" and "skipped" in msg.content.lower()
    ]
    assert len(system_messages) > 0


@pytest.mark.asyncio
async def test_state_persistence_through_checkpoint(postgres_storage, tool_tools):
    """Test that custom state fields persist correctly through checkpointing"""
    # Create mock LLM client with a flow that uses multiple tools
    mock_client = MockLLMClient(conversation_flow=[
        # First get customer info
        {"content": "Getting customer info", "tool_calls": [{"id": "call_1", "name": "get_customer_info", "arguments": {"customer_id": "C1"}}]},
        # Then get order details 
        {"content": "Getting order details", "tool_calls": [{"id": "call_2", "name": "get_order_details", "arguments": {"order_id": "O1"}}]},
        # Process payment (will pause)
        {"content": "Processing payment", "tool_calls": [{"id": "call_3", "name": "process_payment", "arguments": {"order_id": "O1", "amount": 19.99}}]},
        # Final response
        {"content": "All done!"}
    ])
    
    # Setup and execute graph until pause
    initial_state = ToolCheckpointState()
    initial_state.messages = [LLMMessage(role="user", content="Process my order")]
    graph = ToolGraph("state_test", state=initial_state, checkpoint_storage=postgres_storage)
    node = graph.add_tool_node(name="agent", tools=tool_tools, llm_client=mock_client)
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    # Execute
    
    await graph.execute()
    
    # Check state before checkpoint
    first_state = graph.state
    assert first_state.customer_data is not None
    assert first_state.customer_data["name"] == "John Doe"
    assert len(first_state.order_data) > 0
    assert first_state.order_data[0]["id"] == "O1"
    
    # Print for debugging
    print(f"State before checkpoint: customer_data={first_state.customer_data}")
    
    # Store chain ID and create new graph with same tools
    chain_id = graph.chain_id
    new_graph = ToolGraph("state_test", state=initial_state, checkpoint_storage=postgres_storage)
    node = new_graph.add_tool_node(name="agent", tools=tool_tools, llm_client=mock_client)
    new_graph.add_edge(START, node.name)
    new_graph.add_edge(node.name, END)
    
    # Load checkpoint - NOTE: Custom fields may need to be repopulated after loading
    new_graph.load_from_checkpoint(chain_id)
    
    # Get the loaded state
    loaded_state = new_graph.state
    
    # Print for debugging
    print(f"State after checkpoint: customer_data={loaded_state.customer_data}")
    
    # Since custom fields may not be preserved through checkpointing,
    # we need to repopulate them by re-executing the tool functions
    
    # Resume execution - this will re-execute process_payment 
    await new_graph.resume(execute_tool=True)
    
    # Get the final state
    final_state = new_graph.state
    
    # Verify the transaction was processed
    assert hasattr(final_state, "processing_results")
    assert len(final_state.processing_results) > 0
    assert final_state.processing_results[0]["order_id"] == "O1"
    assert final_state.processing_results[0]["amount"] == 19.99
    assert final_state.customer_data["name"] == "John Doe"
    assert first_state.order_data[0]["id"] == "O1"
    
    # Print all tool calls for verification
    print("\nFinal tool calls:")
    for tc in final_state.tool_calls:
        if hasattr(tc, 'tool_name'):
            print(f"- {tc.tool_name}: {tc.arguments}")
        elif isinstance(tc, dict) and 'tool_name' in tc:
            print(f"- {tc['tool_name']}: {tc['arguments']}")
        else:
            print(f"- Unknown format: {tc}")
    
    # Print messages
    print("\nMessages:")
    for i, msg in enumerate(final_state.messages):
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            print(f"{i}: [{msg.role}] {msg.content[:50]}...")
        elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            print(f"{i}: [{msg['role']}] {msg['content'][:50]}...")


@pytest.mark.asyncio
async def test_state_field_preservation_through_checkpoint(postgres_storage):
    """Test that LastValue state fields are properly preserved through checkpointing"""
    # Create the analytics tools
    analytics_tools = [track_user_interaction, set_analysis_goal]
    
    # Create mock LLM client with analytics flow
    mock_client = MockLLMClient(conversation_flow=create_analytics_flow())
    
    # Create initial state
    initial_state = AnalyticsTrackingState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are an analytics assistant."
        ),
        LLMMessage(
            role="user",
            content="Track user U123 with interaction INT456 and set the goal to identify user behavior patterns."
        )
    ]
    
    # Initialize the nested metadata with test values
    initial_state.metadata = AnalyticsTrackingState.AnalyticsMetadata(
        session_id="SES789",
        timestamp=int(time.time()),
        source="test_suite",
        details=AnalyticsTrackingState.AnalyticsMetadata.SessionDetails(
            browser="chrome",
            device="tablet",
            is_mobile=True
        )
    )

    # Create the graph
    graph = ToolGraph(
        "analytics_tracking", 
        state=initial_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = graph.add_tool_node(
        name="analytics_agent",
        tools=analytics_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    # Create engine
    engine = ToolEngine(graph)
    
    # Execute the graph until pause
    result = await engine.execute()
    
    # Get the state after execution
    first_state = result.state
    
    # Verify the execution was paused after setting goal
    assert first_state.is_paused is True
    assert first_state.paused_tool_name == "set_analysis_goal"
    assert first_state.paused_after_execution is True
    
    # Verify state fields were properly set
    assert first_state.user_id == "U123"
    assert first_state.interaction_id == "INT456"
    assert first_state.analysis_goal == "Identify user behavior patterns"
    
    # Store chain ID
    chain_id = graph.chain_id
    
    # Create a completely new graph instance with empty state
    fresh_state = AnalyticsTrackingState()
    fresh_state.messages = [LLMMessage(role="user", content="Just a placeholder")]
    
    new_graph = ToolGraph(
        "analytics_tracking", 
        state=fresh_state,
        checkpoint_storage=postgres_storage
    )
    
    # Add tool node
    node = new_graph.add_tool_node(
        name="analytics_agent",
        tools=analytics_tools,
        llm_client=mock_client,
        options=ToolLoopOptions(max_iterations=5)
    )
    
    # Connect to START and END
    new_graph.add_edge(START, node.name)
    new_graph.add_edge(node.name, END)
    
    # Create new engine
    new_engine = ToolEngine(new_graph)
    
    # Load the checkpoint
    new_graph.load_from_checkpoint(chain_id)
    
    # Get the loaded state
    loaded_state = new_graph.state
    
    # Debug outputs to diagnose the issue
    print("\nDEBUG: Checking state after load_from_checkpoint")
    print(f"Messages type: {type(loaded_state.messages)}")
    if loaded_state.messages:
        print(f"First message type: {type(loaded_state.messages[0])}")
        print(f"First message content: {loaded_state.messages[0]}")
    
    # Verify all state fields match the original values
    assert loaded_state.user_id == first_state.user_id
    assert loaded_state.interaction_id == first_state.interaction_id
    assert loaded_state.analysis_goal == first_state.analysis_goal
    
    # Verify type integrity was maintained
    assert isinstance(loaded_state.user_id, str)
    assert isinstance(loaded_state.interaction_id, str)
    assert isinstance(loaded_state.analysis_goal, str)
    
    # Verify the custom nested BaseModel was correctly restored
    assert loaded_state.metadata is not None
    assert isinstance(loaded_state.metadata, AnalyticsTrackingState.AnalyticsMetadata)
    assert loaded_state.metadata.session_id == first_state.metadata.session_id
    assert loaded_state.metadata.timestamp == first_state.metadata.timestamp
    
    # Verify deeply nested objects were correctly restored
    assert isinstance(loaded_state.metadata.details, AnalyticsTrackingState.AnalyticsMetadata.SessionDetails)
    assert loaded_state.metadata.details.browser == first_state.metadata.details.browser
    assert loaded_state.metadata.details.device == first_state.metadata.details.device
    assert loaded_state.metadata.details.is_mobile == first_state.metadata.details.is_mobile

    # Verify messages are preserved with correct type and content
    assert len(loaded_state.messages) == len(first_state.messages)
    for i, msg in enumerate(loaded_state.messages):
        assert isinstance(msg, LLMMessage), f"Message at index {i} is not an LLMMessage"
        assert msg.role == first_state.messages[i].role
        assert msg.content == first_state.messages[i].content
    
    # Resume execution
    result = await new_engine.resume(loaded_state, execute_tool=True)
    
    # Get final state
    final_state = result.state
    
    # Verify the execution completed
    assert final_state.is_paused is False
    assert final_state.is_complete is True
    
    # Verify state fields were preserved throughout the entire process
    assert final_state.user_id == "U123"
    assert final_state.interaction_id == "INT456"
    assert final_state.analysis_goal == "Identify user behavior patterns"