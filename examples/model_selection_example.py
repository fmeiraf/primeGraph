"""
Example demonstrating how to specify different models for ToolGraph nodes.

This example shows:
1. Setting up a basic ToolGraph with a custom model
2. Using different models for different nodes in the same graph
3. Passing additional API parameters through api_kwargs
"""

import asyncio
import os
from typing import Any, Dict

from dotenv import load_dotenv
from rich import print as rprint

from primeGraph import END, START
from primeGraph.graph import (LLMClientFactory, Provider, ToolGraph,
                              ToolLoopOptions, ToolState, tool)

# Load environment variables for API keys
load_dotenv()


# Define a simple tool for demonstration
@tool("Get greeting")
async def get_greeting(name: str) -> Dict[str, Any]:
    """Get a greeting for a person."""
    return {"greeting": f"Hello, {name}!"}


@tool("Get weather")
async def get_weather(location: str) -> Dict[str, Any]:
    """Get weather for a location (simulated)."""
    # This is just a simulation
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny",
        "forecast": "clear skies"
    }


async def main():
    # Check if we have API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_api_key and not anthropic_api_key:
        print("Error: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        return
    
    # Create clients based on available API keys
    clients = {}
    
    if openai_api_key:
        clients["openai"] = LLMClientFactory.create_client(Provider.OPENAI)
        print("✓ OpenAI client created")
    
    if anthropic_api_key:
        clients["anthropic"] = LLMClientFactory.create_client(Provider.ANTHROPIC)
        print("✓ Anthropic client created")
    
    # Create a graph
    graph = ToolGraph("model_selection_example")
    
    # Example 1: Basic model selection
    if "openai" in clients:
        # Create a node using GPT-3.5 Turbo (cheaper than the default GPT-4)
        options = ToolLoopOptions(
            max_iterations=2,
            model="gpt-3.5-turbo",
            api_kwargs={"temperature": 0.7}
        )
        
        openai_node = graph.add_tool_node(
            name="openai_greeter",
            tools=[get_greeting],
            llm_client=clients["openai"],
            options=options
        )
        
        # Connect to graph
        if not any(edge.start_node == START for edge in graph._all_edges):
            graph.add_edge(START, openai_node.name)
        
        print(f"✓ Added OpenAI node with model: {options.model}")
    
    # Example 2: Using Claude model
    if "anthropic" in clients:
        # Create a node using a specific Claude model
        options = ToolLoopOptions(
            max_iterations=2,
            model="claude-3-haiku-20240307",  # Using the fastest Claude model
            api_kwargs={"temperature": 0.5}
        )
        
        anthropic_node = graph.add_tool_node(
            name="anthropic_weather",
            tools=[get_weather],
            llm_client=clients["anthropic"],
            options=options
        )
        
        # Connect to graph
        if "openai" in clients:
            # If we have an OpenAI node, connect after it
            graph.add_edge(openai_node.name, anthropic_node.name)
        else:
            # Otherwise connect directly to START
            graph.add_edge(START, anthropic_node.name)
        
        print(f"✓ Added Anthropic node with model: {options.model}")
    
    # Connect the last node to END
    last_node = anthropic_node.name if "anthropic" in clients else openai_node.name
    graph.add_edge(last_node, END)
    
    # Create state with initial prompt
    initial_state = ToolState()
    initial_state.messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Get a greeting for Alice and check the weather in San Francisco."
    }]
    
    # Execute the graph
    print("\nExecuting graph...")
    await graph.execute()
    
    # Print final results
    print("\nExecution complete!")
    print("Final state:")
    print(f"- Tool calls: {len(graph.state.tool_calls)}")
    print(f"- Final output: {graph.state.final_output}")
    rprint(graph.state)



if __name__ == "__main__":
    asyncio.run(main()) 