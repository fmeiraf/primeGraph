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
from primeGraph.graph.llm_tools import (LLMMessage, ToolEngine, ToolGraph,
                                        ToolLoopOptions, ToolState, tool)

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
    """Create a conversation flow that uses the payment tool which pauses"""
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
                    "arguments": {"order_id": "O2", "amount": 49.99}
                }
            ]
        },
        # Final response (only reached after resume)
        {
            "content": "The payment for order O2 in the amount of $49.99 has been successfully processed."
        }
    ]


@pytest.fixture
def customer_tools():
    """Fixture providing customer service tools"""
    return [get_customer_info, get_order_details, cancel_order]

@pytest.fixture
def customer_tools_with_payment():
    """Fixture providing customer service tools including the payment tool that pauses"""
    return [get_customer_info, get_order_details, cancel_order, process_payment]


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
    """Fixture providing a mock client for payment scenario with pausing"""
    return MockLLMClient(conversation_flow=create_tool_flow_for_payment())


@pytest.fixture
def tool_graph_with_mock(customer_tools, mock_llm_client_for_cancel):
    """Fixture providing a tool graph with mock client"""
    graph = ToolGraph("customer_service", state_class=CustomerServiceState)
    
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
    """Fixture providing a tool graph with payment processing and pause functionality"""
    graph = ToolGraph("payment_processing", state_class=CustomerServiceState)
    
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
    graph = ToolGraph("openai_customer_service", state_class=CustomerServiceState)
    
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
    graph = ToolGraph("anthropic_customer_service", state_class=CustomerServiceState)
    
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
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available")
async def test_openai_cancel_orders(openai_tool_graph):
    """Test cancelling orders with OpenAI (skipped if no API key)"""
    # Create engine
    engine = ToolEngine(openai_tool_graph)
    
    # Create initial state with request to cancel all orders
    initial_state = CustomerServiceState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise in your responses."
        ),
        LLMMessage(
            role="user",
            content="Please cancel all orders for customer C1."
        )
    ]
    
    # Execute the graph
    result = await engine.execute(initial_state=initial_state)
    
    # Check state
    final_state = result.state
    
    # Verify tool call types are as expected 
    tool_names = [call.tool_name for call in final_state.tool_calls]
    assert len(tool_names) >= 1  # Should make at least one tool call
    
    # It should at least get customer info
    assert "get_customer_info" in tool_names
    
    # Verify all calls succeeded
    assert all(call.success for call in final_state.tool_calls)
    
    cancelled_order_args = [
        call.arguments.get("order_id") 
        for call in final_state.tool_calls 
        if call.tool_name == "cancel_order"
    ]
    
    # If the LLM did decide to cancel orders, make sure it used valid order IDs
    if cancelled_order_args:
        assert any(order_id in ["O1", "O2"] for order_id in cancelled_order_args)
    
    # Verify completion state
    assert final_state.is_complete is True
    assert final_state.final_output is not None


# Test with Anthropic
@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
async def test_anthropic_cancel_orders(anthropic_tool_graph):
    """Test cancelling orders with Anthropic (skipped if no API key)"""
    # Create engine
    engine = ToolEngine(anthropic_tool_graph)
    
    # Create initial state with request to cancel all orders
    initial_state = CustomerServiceState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise in your responses."
        ),
        LLMMessage(
            role="user",
            content="Please cancel all orders for customer C1."
        )
    ]
    
    # Execute the graph
    result = await engine.execute(initial_state=initial_state)
    
    # Check state
    final_state = result.state
    
    # Verify tool call types are as expected 
    tool_names = [call.tool_name for call in final_state.tool_calls]
    assert len(tool_names) >= 1  # Should make at least one tool call
    
    # It should at least get customer info
    assert "get_customer_info" in tool_names
    
    # Verify all calls succeeded
    assert all(call.success for call in final_state.tool_calls)
    
    cancelled_order_args = [
        call.arguments.get("order_id") 
        for call in final_state.tool_calls 
        if call.tool_name == "cancel_order"
    ]
    
    # If the LLM did decide to cancel orders, make sure it used valid order IDs
    if cancelled_order_args:
        assert any(order_id in ["O1", "O2"] for order_id in cancelled_order_args)
    
    # Verify completion state
    assert final_state.is_complete is True
    assert final_state.final_output is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available")
async def test_openai_order_query(openai_client, customer_tools):
    """Test order query with OpenAI (skipped if no API key)"""
    # Create graph
    graph = ToolGraph("openai_order_query", state_class=CustomerServiceState)
    
    node = graph.add_tool_node(
        name="openai_order_query_agent",
        tools=customer_tools,
        llm_client=openai_client,
        options=ToolLoopOptions(max_iterations=3)
    )
    
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    engine = ToolEngine(graph)
    
    # Create initial state with order query
    initial_state = CustomerServiceState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="What's the status of order O2?"
        )
    ]
    
    # Execute the graph
    result = await engine.execute(initial_state=initial_state)
    
    # Check state
    final_state = result.state
    
    # Check if the LLM made any tool calls
    tool_names = [call.tool_name for call in final_state.tool_calls]
    
    # If there were tool calls, they should include get_order_details
    if tool_names:
        assert "get_order_details" in tool_names
        
        # Find the get_order_details call for O2
        order_query_calls = [
            call for call in final_state.tool_calls
            if call.tool_name == "get_order_details" and call.arguments.get("order_id") == "O2"
        ]
        
        # Should have at least one such call
        assert len(order_query_calls) >= 1
        assert order_query_calls[0].success is True
    
    # Verify completion state
    assert final_state.is_complete is True


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
async def test_anthropic_order_query(anthropic_client, customer_tools):
    """Test order query with Anthropic (skipped if no API key)"""
    # Create graph
    graph = ToolGraph("anthropic_order_query", state_class=CustomerServiceState)
    
    node = graph.add_tool_node(
        name="anthropic_order_query_agent",
        tools=customer_tools,
        llm_client=anthropic_client,
        options=ToolLoopOptions(max_iterations=3)
    )
    
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    engine = ToolEngine(graph)
    
    # Create initial state with order query
    initial_state = CustomerServiceState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="What's the status of order O2?"
        )
    ]
    
    # Execute the graph
    result = await engine.execute(initial_state=initial_state)
    
    # Check state
    final_state = result.state
    
    # Check if the LLM made any tool calls
    tool_names = [call.tool_name for call in final_state.tool_calls]
    
    # If there were tool calls, they should include get_order_details
    if tool_names:
        assert "get_order_details" in tool_names
        
        # Find the get_order_details call for O2
        order_query_calls = [
            call for call in final_state.tool_calls
            if call.tool_name == "get_order_details" and call.arguments.get("order_id") == "O2"
        ]
        
        # Should have at least one such call
        assert len(order_query_calls) >= 1
        assert order_query_calls[0].success is True
    
    # Verify completion state
    assert final_state.is_complete is True