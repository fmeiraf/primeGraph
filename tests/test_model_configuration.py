"""
Tests for model configuration in ToolGraph.

These tests verify that:
1. Different models can be specified in ToolLoopOptions
2. The model parameter is correctly passed to the LLM client
3. Both OpenAI and Anthropic clients handle the model parameter correctly
"""

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from pydantic import Field

from primeGraph.buffer.factory import History
from primeGraph.constants import END, START
from primeGraph.graph.llm_clients import (AnthropicClient, LLMClientFactory,
                                          OpenAIClient, Provider)
from primeGraph.graph.llm_tools import (LLMMessage, ToolGraph, ToolLoopOptions,
                                        ToolState, tool)

load_dotenv()


class ModelTestState(ToolState):
    """State for model configuration testing"""
    test_results: History[Dict[str, Any]] = Field(default_factory=list)


# Define a simple test tool
@tool("Echo input")
async def echo_tool(input_text: str) -> Dict[str, Any]:
    """Echo back the input text"""
    return {"result": input_text}


class MockOpenAIClient(OpenAIClient):
    """Mock OpenAI client that tracks the model parameter"""
    
    def __init__(self):
        """Initialize the mock client"""
        super().__init__(api_key="mock_key")
        self.last_kwargs = {}
        self.call_count = 0
        
    async def generate(self, messages, tools=None, tool_choice=None, **kwargs):
        """Record kwargs and return a simple response"""
        self.last_kwargs = kwargs
        self.call_count += 1
        
        # Create a simple mock response that doesn't use tools
        content = "This is a test response"
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.choices[0].message.tool_calls = None
        
        return content, response


class MockAnthropicClient(AnthropicClient):
    """Mock Anthropic client that tracks the model parameter"""
    
    def __init__(self):
        """Initialize the mock client"""
        super().__init__(api_key="mock_key")
        self.last_kwargs = {}
        self.call_count = 0
        
    async def generate(self, messages, tools=None, tool_choice=None, **kwargs):
        """Record kwargs and return a simple response"""
        self.last_kwargs = kwargs
        self.call_count += 1
        
        # Create a simple mock response
        content = "This is a test response"
        response = MagicMock()
        response.content = [MagicMock(type="text", text=content)]
        
        return content, response


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mock OpenAI client"""
    return MockOpenAIClient()


@pytest.fixture
def mock_anthropic_client():
    """Fixture providing a mock Anthropic client"""
    return MockAnthropicClient()


def create_test_graph(llm_client, model=None, api_kwargs=None):
    """Create a test graph with the specified model configuration"""
    # Create state instance
    state = ModelTestState()
    
    # Create graph with state instance
    graph = ToolGraph("model_test", state=state)
    
    options = ToolLoopOptions(
        max_iterations=2,
        max_tokens=1024
    )
    
    # Set model if provided
    if model:
        options.model = model
    
    # Set additional kwargs if provided
    if api_kwargs:
        options.api_kwargs = api_kwargs
    
    node = graph.add_tool_node(
        name="test_agent",
        tools=[echo_tool],
        llm_client=llm_client,
        options=options
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    return graph


def create_initial_state():
    """Create an initial state for testing"""
    initial_state = ModelTestState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a test assistant."
        ),
        LLMMessage(
            role="user",
            content="This is a test message."
        )
    ]
    return initial_state


@pytest.mark.asyncio
async def test_openai_model_configuration(mock_openai_client):
    """Test that OpenAI model configuration works correctly"""
    # Test with default model
    default_graph = create_test_graph(mock_openai_client)
    
    # Set the state directly on the graph
    default_graph.state = create_initial_state()
    
    # Use the graph directly instead of creating a new engine
    await default_graph.execute()
    
    # The default model should not be specified (library default will be used)
    assert "model" not in mock_openai_client.last_kwargs or mock_openai_client.last_kwargs["model"] == "gpt-4-turbo"
    
    # Reset call count
    mock_openai_client.call_count = 0
    
    # Test with specific model
    custom_model = "gpt-3.5-turbo"
    custom_graph = create_test_graph(mock_openai_client, model=custom_model)
    
    # Set the state directly on the graph
    custom_graph.state = create_initial_state()
    
    # Use the graph directly instead of creating a new engine
    await custom_graph.execute()
    
    # The specified model should be used
    assert mock_openai_client.last_kwargs.get("model") == custom_model
    assert mock_openai_client.call_count == 1


@pytest.mark.asyncio
async def test_anthropic_model_configuration(mock_anthropic_client):
    """Test that Anthropic model configuration works correctly"""
    # Test with default model
    default_graph = create_test_graph(mock_anthropic_client)
    
    # Set the state directly on the graph
    default_graph.state = create_initial_state()
    
    # Use the graph directly
    await default_graph.execute()
    
    # The default model should not be specified (library default will be used)
    assert "model" not in mock_anthropic_client.last_kwargs or mock_anthropic_client.last_kwargs["model"] == "claude-3-opus-20240229"
    
    # Reset call count
    mock_anthropic_client.call_count = 0
    
    # Test with specific model
    custom_model = "claude-3-sonnet-20240229"
    custom_graph = create_test_graph(mock_anthropic_client, model=custom_model)
    
    # Set the state directly on the graph
    custom_graph.state = create_initial_state()
    
    # Use the graph directly
    await custom_graph.execute()
    
    # The specified model should be used
    assert mock_anthropic_client.last_kwargs.get("model") == custom_model
    assert mock_anthropic_client.call_count == 1


@pytest.mark.asyncio
async def test_api_kwargs_configuration(mock_openai_client):
    """Test that additional API kwargs are properly passed through"""
    # Test with additional kwargs
    api_kwargs = {
        "temperature": 0.2,
        "top_p": 0.95,
        "frequency_penalty": 0.5
    }
    
    custom_graph = create_test_graph(
        mock_openai_client, 
        model="gpt-4-turbo", 
        api_kwargs=api_kwargs
    )
    
    # Set the state directly on the graph
    custom_graph.state = create_initial_state()
    
    # Execute the graph directly
    await custom_graph.execute()
    
    # Check that all kwargs were properly passed
    for key, value in api_kwargs.items():
        assert mock_openai_client.last_kwargs.get(key) == value
    
    # Also verify the model was set correctly
    assert mock_openai_client.last_kwargs.get("model") == "gpt-4-turbo"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available")
async def test_real_openai_model_selection():
    """Test with actual OpenAI API (skipped if no API key)"""
    # Create a real OpenAI client
    openai_client = LLMClientFactory.create_client(Provider.OPENAI)
    
    # Test with a specific model
    custom_model = "gpt-3.5-turbo"  # Using a cheaper model for testing
    graph = create_test_graph(openai_client, model=custom_model)
    
    # Set the state directly on the graph
    graph.state = create_initial_state()
    
    # Use patch to check if the right model is being passed
    with patch('openai.resources.chat.completions.Completions.create') as mock_create:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_create.return_value = mock_response
        
        # Execute the graph directly
        await graph.execute()
        
        # Check that the model was correctly passed
        call_args = mock_create.call_args[1]  # Get the kwargs of the call
        assert call_args.get('model') == custom_model


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
async def test_real_anthropic_model_selection():
    """Test with actual Anthropic API (skipped if no API key)"""
    # Create a real Anthropic client
    anthropic_client = LLMClientFactory.create_client(Provider.ANTHROPIC)
    
    # Test with a specific model
    custom_model = "claude-3-haiku-20240307"  # Using a cheaper model for testing
    graph = create_test_graph(anthropic_client, model=custom_model)
    
    # Set the state directly on the graph
    graph.state = create_initial_state()
    
    # Use patch to check if the right model is being passed
    with patch('anthropic.resources.messages.Messages.create') as mock_create:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Test response")]
        mock_create.return_value = mock_response
        
        # Execute the graph directly
        await graph.execute()
        
        # Check that the model was correctly passed
        call_args = mock_create.call_args[1]  # Get the kwargs of the call
        assert call_args.get('model') == custom_model 