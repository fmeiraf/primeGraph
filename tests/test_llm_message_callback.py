"""
Tests for the LLM message callback functionality.

These tests verify that the on_message callback is properly triggered when:
1. LLM generates a message with tool calls
2. LLM generates a final message (without tool calls)
"""

import json
import os
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


class MessageCollectorState(ToolState):
    """State for message collector testing"""
    calls_to_on_message: History[Dict[str, Any]] = Field(default_factory=list)
    customer_data: LastValue[Optional[Dict[str, Any]]] = None


# Define a simple tool for testing
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


def create_mock_flow_with_tool_and_final():
    """Create a conversation flow that uses tools first and ends with a normal message"""
    return [
        # First get customer info (tool use)
        {
            "content": "I'll help you get information for customer C1.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_customer_info",
                    "arguments": {"customer_id": "C1"}
                }
            ]
        },
        # Final message (no tool use)
        {
            "content": "Customer John Doe has email john@example.com and 2 orders: O1 and O2."
        }
    ]


@pytest.fixture
def customer_tools():
    """Fixture providing customer info tool"""
    return [get_customer_info]


@pytest.fixture
def mock_llm_client():
    """Fixture providing a mock client for testing message callbacks"""
    return MockLLMClient(conversation_flow=create_mock_flow_with_tool_and_final())


@pytest.fixture
def message_collector_callback():
    """
    Create a callback function that collects all messages received.
    This is used to test that on_message is correctly called.
    """
    messages_received = []
    
    def collector(message_data):
        messages_received.append(message_data)
        print(f"Message collector received: {json.dumps(message_data, default=str)}")
    
    collector.messages = messages_received
    return collector


@pytest.mark.asyncio
async def test_on_message_callback_with_mock(customer_tools, mock_llm_client, message_collector_callback):
    """Test that on_message callback is properly triggered with mock LLM"""
    # Create graph
    # Create initial state with request to get customer info
    initial_state = MessageCollectorState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="Get information for customer C1."
        )
    ]
    graph = ToolGraph("message_collector", state=initial_state)
    
    # Add tool node with on_message callback
    node = graph.add_tool_node(
        name="message_agent",
        tools=customer_tools,
        llm_client=mock_llm_client,
        options=ToolLoopOptions(max_iterations=5),
        on_message=message_collector_callback  # Pass the callback
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    
    
    # Execute the graph directly
    await graph.execute()
    
    
    # Verify the callback was called for all messages (assistant, tool, and final)
    assert len(message_collector_callback.messages) == 3
    
    # Check the first message (assistant with tool calls)
    assistant_message = message_collector_callback.messages[0]
    assert assistant_message["message_type"] == "assistant"
    assert "I'll help you get information for customer C1." in assistant_message["content"]
    assert assistant_message["has_tool_calls"] is True
    # We won't check the tool_calls field directly since our implementation no longer includes it
    assert assistant_message["is_final"] is False
    
    # Check the second message (tool response)
    tool_message = message_collector_callback.messages[1]
    assert tool_message["message_type"] == "tool"
    assert tool_message["tool_name"] == "get_customer_info"
    assert "John Doe" in tool_message["content"]
    assert tool_message["is_final"] is False
    
    # Check the third message (final assistant response)
    final_message = message_collector_callback.messages[2]
    assert final_message["message_type"] == "assistant"
    assert "Customer John Doe has email" in final_message["content"]
    assert final_message["has_tool_calls"] is False
    assert final_message["is_final"] is True


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available")
async def test_on_message_callback_with_openai(customer_tools, message_collector_callback):
    """Test on_message callback with real OpenAI client"""
    # Get OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available for OpenAI LLM test")
        
    openai_client = LLMClientFactory.create_client(Provider.OPENAI, api_key=api_key)
    
    # Create graph
    # Create initial state with request to get customer info
    initial_state = MessageCollectorState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Be concise."
        ),
        LLMMessage(
            role="user",
            content="Get information for customer C1."
        )
    ]
    graph = ToolGraph("openai_message_collector", state=initial_state)
    
    # Add tool node with on_message callback
    node = graph.add_tool_node(
        name="openai_message_agent",
        tools=customer_tools,
        llm_client=openai_client,
        options=ToolLoopOptions(max_iterations=5),
        on_message=message_collector_callback  # Pass the callback
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    
    
    
    # Execute the graph directly
    await graph.execute()
    
    
    
    # Verify the callback was called at least once
    assert len(message_collector_callback.messages) >= 1
    
    # With OpenAI, it might choose different approaches, but verify the key structure
    for message in message_collector_callback.messages:
        assert "message_type" in message
        assert message["message_type"] in ["assistant", "tool", "system"]  # Allow any valid message type
        assert "content" in message
        # Check other fields based on message type
        if message["message_type"] == "system":
            # System messages should have error information for OpenAI errors
            assert "is_final" in message
            assert "is_error" in message
        elif message["message_type"] == "assistant":
            # Assistant messages should have standard fields
            assert "content" in message
            # Note: We don't check for tool_calls directly as it might not be in the message object
            # even when has_tool_calls is True - the raw_response contains this information instead
        elif message["message_type"] == "tool":
            # Tool messages should have tool information
            assert "tool_name" in message or "tool_id" in message

    # The last message should be final
    assert message_collector_callback.messages[-1]["is_final"] is True