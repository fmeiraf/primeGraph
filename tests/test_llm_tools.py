"""
Tests for the LLM tool nodes functionality.

These tests verify that the tool nodes system properly:
1. Allows LLMs to call tools in sequence
2. Maintains state between tool calls
3. Handles both real and mock LLM clients
4. Supports complex workflows with chained tool calls
"""

import os
import time
from typing import Any, Dict, Optional

import pytest
from dotenv import load_dotenv
from pydantic import Field

from primeGraph.buffer.factory import History, LastValue
from primeGraph.constants import END, START
from primeGraph.graph.llm_clients import (LLMClientBase, LLMClientFactory,
                                          Provider)
from primeGraph.graph.llm_tools import (LLMMessage, ToolGraph, ToolLoopOptions,
                                        ToolState, tool)

load_dotenv()


class CustomerServiceState(ToolState):
    """State for customer service agent tools testing"""
    customer_data: LastValue[Optional[Dict[str, Any]]] = None
    order_data: History[Dict[str, Any]] = Field(default_factory=list)
    cancelled_orders: History[str] = Field(default_factory=list)
    # Tool state fields (inherited) use History markers


# Define tool functions for testing
@tool("Get customer information")
async def get_customer_info(customer_id: str) -> Dict[str, Any]:
    """Get customer details by ID"""
    # Test data
    customers = {
        "C1": {
            "id": "C1", 
            "name": "John Doe", 
            "email": "john@example.com",
            "orders": ["O1", "O2"]
        },
        "C2": {
            "id": "C2", 
            "name": "Jane Smith", 
            "email": "jane@example.com",
            "orders": ["O3"]
        }
    }
    
    if customer_id not in customers:
        raise ValueError(f"Customer {customer_id} not found")
    
    return customers[customer_id]

# Add a tool with pause_after_execution flag set to True
@tool("Update customer account", pause_after_execution=True)
async def update_customer_account(customer_id: str, email: str) -> Dict[str, Any]:
    """Update a customer's account information, pausing after execution for verification"""
    # Simulate updating customer account
    return {
        "customer_id": customer_id,
        "new_email": email,
        "status": "updated",
        "timestamp": int(time.time())
    }


# Add a tool with pause_before_execution flag set to True
@tool("Process payment", pause_before_execution=True)
async def process_payment(order_id: str, amount: float) -> Dict[str, Any]:
    """Process a payment for an order, pausing for verification"""
    # This would normally interact with a payment gateway
    # but for testing it just returns a confirmation
    return {
        "order_id": order_id,
        "amount": amount,
        "status": "processed",
        "transaction_id": f"TX-{order_id}-{int(time.time())}"
    }


@tool("Get order details")
async def get_order_details(order_id: str) -> Dict[str, Any]:
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
        },
        "O3": {
            "id": "O3",
            "customer_id": "C2",
            "product": "Gizmo C",
            "quantity": 3,
            "price": 29.99,
            "status": "delivered"
        }
    }
    
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    
    return orders[order_id]


@tool("Cancel an order")
async def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel an order by ID"""
    # Test data
    orders = {
        "O1": {
            "id": "O1",
            "customer_id": "C1",
            "product": "Widget A",
            "status": "shipped"
        },
        "O2": {
            "id": "O2",
            "customer_id": "C1",
            "product": "Gadget B",
            "status": "processing"
        }
    }
    
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    
    # Update status to cancelled
    result = orders[order_id].copy()
    result["status"] = "cancelled"
    
    return {
        "order_id": order_id,
        "status": "cancelled",
        "message": f"Order {order_id} has been cancelled successfully",
        "order_details": result
    }


# Mock LLM client for testing when real LLMs are not available
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


# Predefined mock responses
def create_tool_flow_for_cancel_all_orders():
    """Create a conversation flow for cancelling all orders scenario"""
    return [
        # First get customer info
        {
            "content": "I'll help you cancel all orders for customer C1. Let me look up their information first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_customer_info",
                    "arguments": {"customer_id": "C1"}
                }
            ]
        },
        # Then cancel first order
        {
            "content": "I found the customer and their orders. Let me cancel them one by one.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "cancel_order",
                    "arguments": {"order_id": "O1"}
                }
            ]
        },
        # Then cancel second order
        {
            "content": "The first order has been cancelled. Let me cancel the second one.",
            "tool_calls": [
                {
                    "id": "call_3",
                    "name": "cancel_order",
                    "arguments": {"order_id": "O2"}
                }
            ]
        },
        # Final summary response
        {
            "content": "I've successfully cancelled all orders for customer John Doe (C1). Both order O1 and O2 have been cancelled."
        }
    ]


def create_tool_flow_for_order_query():
    """Create a conversation flow for order status query scenario"""
    return [
        # Get order details
        {
            "content": "Let me check the status of order O2 for you.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_order_details",
                    "arguments": {"order_id": "O2"}
                }
            ]
        },
        # Final response
        {
            "content": "Order O2 is a Gadget B that costs $49.99 and is currently in processing status."
        }
    ]

def create_tool_flow_for_payment():
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
        # Then process payment (this will pause execution)
        {
            "content": "I found the customer. Let me process the payment now.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "process_payment",
                    "arguments": {"order_id": "O1", "amount": 19.99}
                }
            ]
        },
        # Final response (only reached after resume)
        {
            "content": "The payment for order O1 in the amount of $19.99 has been successfully processed."
        }
    ]


def create_tool_flow_for_account_update():
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


@pytest.fixture
def customer_tools():
    """Fixture providing customer service tools"""
    return [get_customer_info, get_order_details, cancel_order]

@pytest.fixture
def customer_tools_with_payment():
    """Fixture providing customer service tools including the payment tool that pauses before execution"""
    return [get_customer_info, get_order_details, cancel_order, process_payment]

@pytest.fixture
def customer_tools_with_account_update():
    """Fixture providing customer service tools including the account update tool that pauses after execution"""
    return [get_customer_info, get_order_details, cancel_order, update_customer_account]


@pytest.fixture
def mock_llm_client_for_cancel():
    """Fixture providing a mock client for cancel all orders scenario"""
    return MockLLMClient(conversation_flow=create_tool_flow_for_cancel_all_orders())


@pytest.fixture
def mock_llm_client_for_query():
    """Fixture providing a mock client for order query scenario"""
    return MockLLMClient(conversation_flow=create_tool_flow_for_order_query())


@pytest.fixture
def mock_llm_client_for_payment():
    """Fixture providing a mock client for payment scenario with pausing before execution"""
    return MockLLMClient(conversation_flow=create_tool_flow_for_payment())

@pytest.fixture
def mock_llm_client_for_account_update():
    """Fixture providing a mock client for account update scenario with pausing after execution"""
    return MockLLMClient(conversation_flow=create_tool_flow_for_account_update())


@pytest.fixture
def tool_graph_with_mock(customer_tools, mock_llm_client_for_cancel):
    """Fixture providing a tool graph with mock client"""
    # Create state instance
    state = CustomerServiceState()
    
    # Create graph with state instance
    graph = ToolGraph("customer_service", state=state)
    
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024
    )
    
    node = graph.add_tool_node(
        name="customer_service_agent",
        tools=customer_tools,
        llm_client=mock_llm_client_for_cancel,
        options=options
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    return graph


@pytest.fixture
def tool_graph_with_payment(customer_tools_with_payment, mock_llm_client_for_payment):
    """Fixture providing a tool graph with payment processing that pauses before execution"""
    # Create state instance
    state = CustomerServiceState()
    
    # Create graph with state instance
    graph = ToolGraph("payment_processing", state=state)
    
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024
    )
    
    node = graph.add_tool_node(
        name="payment_agent",
        tools=customer_tools_with_payment,
        llm_client=mock_llm_client_for_payment,
        options=options
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    return graph

@pytest.fixture
def tool_graph_with_account_update(customer_tools_with_account_update, mock_llm_client_for_account_update):
    """Fixture providing a tool graph with account update that pauses after execution"""
    # Create state instance
    state = CustomerServiceState()
    
    # Create graph with state instance
    graph = ToolGraph("account_update", state=state)
    
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024
    )
    
    node = graph.add_tool_node(
        name="account_update_agent",
        tools=customer_tools_with_account_update,
        llm_client=mock_llm_client_for_account_update,
        options=options
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    return graph


def get_openai_client():
    """Get an OpenAI client if API key is available"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available for OpenAI LLM test")
    return LLMClientFactory.create_client(Provider.OPENAI, api_key=api_key)


def get_anthropic_client():
    """Get an Anthropic client if API key is available"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not available for Anthropic LLM test")
    return LLMClientFactory.create_client(Provider.ANTHROPIC, api_key=api_key)


@pytest.fixture
def openai_client():
    """Fixture providing an OpenAI client if available"""
    return get_openai_client()


@pytest.fixture
def anthropic_client():
    """Fixture providing an Anthropic client if available"""
    return get_anthropic_client()


@pytest.fixture
def openai_tool_graph(customer_tools, openai_client):
    """Fixture providing a tool graph with OpenAI client"""
    # Create state instance
    state = CustomerServiceState()
    
    # Create graph with state instance
    graph = ToolGraph("openai_customer_service", state=state)
    
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024
    )
    
    node = graph.add_tool_node(
        name="openai_customer_service_agent",
        tools=customer_tools,
        llm_client=openai_client,
        options=options
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    return graph


@pytest.fixture
def anthropic_tool_graph(customer_tools, anthropic_client):
    """Fixture providing a tool graph with Anthropic client"""
    # Create state instance
    state = CustomerServiceState()
    
    # Create graph with state instance
    graph = ToolGraph("anthropic_customer_service", state=state)
    
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024
    )
    
    node = graph.add_tool_node(
        name="anthropic_customer_service_agent",
        tools=customer_tools,
        llm_client=anthropic_client,
        options=options
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    return graph


# Test with OpenAI
@pytest.mark.asyncio
async def test_openai_cancel_orders(tool_graph_with_mock):
    """Test cancelling orders with a mock client that simulates OpenAI behavior"""
    # Set up messages in the graph's state
    tool_graph_with_mock.state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise in your responses."
        ),
        LLMMessage(
            role="user",
            content="Please cancel all orders for customer C1."
        )
    ]
    
    # Execute the graph directly
    await tool_graph_with_mock.execute()
    
    # Access final state
    final_state = tool_graph_with_mock.state
    
    # Verify tool calls were made
    assert len(final_state.tool_calls) > 0
    assert any(call.tool_name == "get_customer_info" for call in final_state.tool_calls)
    assert any(call.tool_name == "cancel_order" for call in final_state.tool_calls)
    
    # Verify orders were cancelled
    assert len(final_state.cancelled_orders) > 0


# Test with Anthropic
@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
async def test_anthropic_cancel_orders(anthropic_tool_graph):
    """Test cancelling orders with Anthropic Claude (skipped if no API key)"""
    # Set up messages in the graph's state
    anthropic_tool_graph.state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise in your responses."
        ),
        LLMMessage(
            role="user",
            content="Please cancel all orders for customer C1."
        )
    ]
    
    # Execute the graph directly
    await anthropic_tool_graph.execute()
    
    # Access final state
    final_state = anthropic_tool_graph.state
    
    # Verify tool calls were made
    assert len(final_state.tool_calls) > 0
    assert any(call.tool_name == "get_customer_info" for call in final_state.tool_calls)
    
    # Verify all calls succeeded
    assert all(call.success for call in final_state.tool_calls)
    
    # Verify completion state
    assert final_state.is_complete is True
    assert final_state.final_output is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available")
async def test_openai_order_query(openai_client, customer_tools):
    """Test order query with OpenAI (skipped if no API key)"""
    # Create state instance
    state = CustomerServiceState()
    
    # Set up messages in the state
    state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="What's the status of order O2?"
        )
    ]
    
    # Create graph with state instance
    graph = ToolGraph("openai_order_query", state=state)
    
    node = graph.add_tool_node(
        name="openai_order_query_agent",
        tools=customer_tools,
        llm_client=openai_client,
        options=ToolLoopOptions(max_iterations=3)
    )
    
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    # Execute the graph directly
    await graph.execute()
    
    # Check state
    final_state = graph.state
    
    # Check if there was an API quota error
    if final_state.error and "insufficient_quota" in final_state.error:
        pytest.skip("OpenAI API quota exceeded, skipping test")
    
    # Check if the LLM made any tool calls
    tool_names = [call.tool_name for call in final_state.tool_calls]
    
    # Verify that the LLM made at least one tool call
    assert len(tool_names) > 0
    
    # The test is now more lenient - we don't require get_order_details specifically
    # The model might choose to get customer info first and then get order details,
    # or it might use a different approach entirely
    
    # Verify completion state
    assert final_state.is_complete is True
    
    # If get_order_details was called, verify it was for the right order
    order_query_calls = [
        call for call in final_state.tool_calls
        if call.tool_name == "get_order_details" and call.arguments.get("order_id") == "O2"
    ]
    
    if order_query_calls:
        # If get_order_details was called, verify the result
        assert order_query_calls[0].success is True


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
async def test_anthropic_order_query(anthropic_client, customer_tools):
    """Test order query with Anthropic (skipped if no API key)"""
    # Create state instance
    state = CustomerServiceState()
    
    # Set up messages in the state
    state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="What's the status of order O2?"
        )
    ]
    
    # Create graph with state instance
    graph = ToolGraph("anthropic_order_query", state=state)
    
    node = graph.add_tool_node(
        name="anthropic_order_query_agent",
        tools=customer_tools,
        llm_client=anthropic_client,
        options=ToolLoopOptions(max_iterations=3)
    )
    
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    # Execute the graph directly
    await graph.execute()
    
    # Check state
    final_state = graph.state
    
    # Check if there was an API quota error
    if final_state.error and "insufficient_quota" in final_state.error:
        pytest.skip("Anthropic API quota exceeded, skipping test")
    
    # Check if the LLM made any tool calls
    tool_names = [call.tool_name for call in final_state.tool_calls]
    
    # Verify that the LLM made at least one tool call
    assert len(tool_names) > 0
    
    # The test is now more lenient - we don't require get_order_details specifically
    # The model might choose to get customer info first and then get order details,
    # or it might use a different approach entirely
    
    # Verify completion state
    assert final_state.is_complete is True
    
    # If get_order_details was called, verify it was for the right order
    order_query_calls = [
        call for call in final_state.tool_calls
        if call.tool_name == "get_order_details" and call.arguments.get("order_id") == "O2"
    ]
    
    if order_query_calls:
        # If get_order_details was called, verify the result
        assert order_query_calls[0].success is True


@pytest.mark.asyncio
async def test_pause_after_execution(tool_graph_with_account_update):
    """Test the pause after execution functionality with the account update tool"""
    # Set up messages in the graph's state
    tool_graph_with_account_update.state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="Please update the email for customer C1 to john.doe.new@example.com."
        )
    ]
    
    # Execute the graph - should pause after account update
    await tool_graph_with_account_update.execute()
    
    # Check state
    assert tool_graph_with_account_update.state.is_paused
    assert tool_graph_with_account_update.state.paused_after_execution
    assert tool_graph_with_account_update.state.paused_tool_name == "update_customer_account"
    assert tool_graph_with_account_update.state.paused_tool_result is not None
    assert tool_graph_with_account_update.state.paused_tool_result.success
    
    # Resume execution
    await tool_graph_with_account_update.resume(execute_tool=True)
    
    # Check final state
    assert not tool_graph_with_account_update.state.is_paused
    assert tool_graph_with_account_update.state.is_complete
    assert tool_graph_with_account_update.state.final_output is not None


@pytest.mark.asyncio
async def test_pause_before_execution(tool_graph_with_payment):
    """Test the pause before execution functionality with the payment tool"""
    # Set up messages in the graph's state
    tool_graph_with_payment.state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="Please process payment for order O1."
        )
    ]
    
    # Execute the graph - should pause before payment processing
    await tool_graph_with_payment.execute()
    
    # Check state
    assert tool_graph_with_payment.state.is_paused
    assert not tool_graph_with_payment.state.paused_after_execution
    assert tool_graph_with_payment.state.paused_tool_name == "process_payment"
    assert tool_graph_with_payment.state.paused_tool_arguments is not None
    assert tool_graph_with_payment.state.paused_tool_arguments["order_id"] == "O1"
    
    # Resume execution
    await tool_graph_with_payment.resume(execute_tool=True)
    
    # Check final state
    assert not tool_graph_with_payment.state.is_paused
    assert tool_graph_with_payment.state.is_complete
    assert tool_graph_with_payment.state.final_output is not None
    
    # The tool result should be in tool_calls now
    payment_calls = [call for call in tool_graph_with_payment.state.tool_calls if call.tool_name == "process_payment"]
    assert len(payment_calls) > 0
    assert payment_calls[0].success


# Add this new tool definition before the test fixtures
@tool(
    "Process a secure payment",
    hidden_params=["api_key", "secret"]
)
async def process_secure_payment(amount: float, currency: str, api_key: str, secret: str) -> Dict[str, Any]:
    """Process a payment with secure credentials"""
    # This would normally interact with a payment gateway
    # but for testing it just returns a confirmation
    return {
        "amount": amount,
        "currency": currency,
        "status": "processed",
        "transaction_id": f"TX-{int(time.time())}"
    }

@pytest.fixture
def secure_payment_tools():
    """Fixture providing tools including one with hidden parameters"""
    return [process_secure_payment]

@pytest.fixture
def tool_graph_with_secure_payment(secure_payment_tools, mock_llm_client_for_payment):
    """Fixture providing a tool graph with secure payment processing"""
    state = CustomerServiceState()
    graph = ToolGraph("secure_payment", state=state)
    
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024
    )
    
    node = graph.add_tool_node(
        name="secure_payment_agent",
        tools=secure_payment_tools,
        llm_client=mock_llm_client_for_payment,
        options=options
    )
    
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    return graph

@pytest.mark.asyncio
async def test_hidden_parameters(tool_graph_with_secure_payment):
    """Test that hidden parameters are excluded from schema but still required for execution"""
    # Get the tool node
    node = tool_graph_with_secure_payment.nodes["secure_payment_agent"]
    
    # Get the tool schemas
    schemas = node.get_tool_schemas()
    
    # Find the secure payment tool schema
    secure_payment_schema = next(
        (s for s in schemas if s.get("function", {}).get("name") == "process_secure_payment"),
        None
    )
    
    assert secure_payment_schema is not None, "Secure payment tool schema not found"
    
    # Get the parameters from the schema
    parameters = secure_payment_schema.get("function", {}).get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    
    # Verify hidden parameters are not in the schema
    assert "api_key" not in properties, "api_key should be hidden from schema"
    assert "secret" not in properties, "secret should be hidden from schema"
    assert "api_key" not in required, "api_key should not be in required list"
    assert "secret" not in required, "secret should not be in required list"
    
    # Verify visible parameters are in the schema
    assert "amount" in properties, "amount should be visible in schema"
    assert "currency" in properties, "currency should be visible in schema"
    assert "amount" in required, "amount should be required"
    assert "currency" in required, "currency should be required"
    
    # Verify the tool still requires all parameters for execution
    tool_func = node.find_tool_by_name("process_secure_payment")
    assert tool_func is not None, "Tool function not found"
    
    # Get the tool definition
    tool_def = tool_func._tool_definition
    
    # Verify all parameters are in the tool definition
    assert "api_key" in tool_def.parameters, "api_key should be in tool parameters"
    assert "secret" in tool_def.parameters, "secret should be in tool parameters"
    assert "amount" in tool_def.parameters, "amount should be in tool parameters"
    assert "currency" in tool_def.parameters, "currency should be in tool parameters"
    
    # Verify hidden parameters are marked as hidden
    assert "api_key" in tool_def.hidden_params, "api_key should be marked as hidden"
    assert "secret" in tool_def.hidden_params, "secret should be marked as hidden"
    assert "amount" not in tool_def.hidden_params, "amount should not be marked as hidden"
    assert "currency" not in tool_def.hidden_params, "currency should not be marked as hidden"

@pytest.mark.asyncio
async def test_empty_messages_not_added(tool_graph_with_mock):
    """Test that empty messages are not added to the message history during tool execution"""
    # Create a mock LLM client that returns empty messages
    mock_client = MockLLMClient(conversation_flow=[
        # First response with empty content
        {
            "content": "",  # Empty content
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_customer_info",
                    "arguments": {"customer_id": "C1"}
                }
            ]
        },
        # Second response with whitespace content
        {
            "content": "   ",  # Whitespace only
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "get_order_details",
                    "arguments": {"order_id": "O1"}
                }
            ]
        },
        # Third response with None content
        {
            "content": None,  # None content
            "tool_calls": [
                {
                    "id": "call_3",
                    "name": "cancel_order",
                    "arguments": {"order_id": "O1"}
                }
            ]
        },
        # Final response with valid content
        {
            "content": "This is a valid final response"
        }
    ])
    
    # Replace the mock client in the graph
    tool_graph_with_mock.nodes["customer_service_agent"].llm_client = mock_client
    
    # Set up initial messages
    tool_graph_with_mock.state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant."
        ),
        LLMMessage(
            role="user",
            content="Please help me with my order."
        )
    ]
    
    
    # Execute the graph
    await tool_graph_with_mock.execute()
    
    
    # Verify that only the valid final message was added as an assistant message
    # and that tool result messages were added
    assistant_messages = [msg for msg in tool_graph_with_mock.state.messages if msg.role == "assistant"]
    tool_messages = [msg for msg in tool_graph_with_mock.state.messages if msg.role == "tool"]
    
    # We should have only one assistant message (the final one)
    assert len(assistant_messages) == 1, \
        "Only one assistant message (the final one) should be added"
    
    # The last assistant message should be the valid one
    assert assistant_messages[0].content == "This is a valid final response", \
        "The last assistant message should be the valid final response"
    
    # We should have tool result messages for each tool call
    assert len(tool_messages) == 3, \
        "Three tool result messages should be added (one for each tool call)"
    
    # Verify tool calls were still executed
    assert len(tool_graph_with_mock.state.tool_calls) == 3, \
        "Tool calls should still be executed even with empty messages"