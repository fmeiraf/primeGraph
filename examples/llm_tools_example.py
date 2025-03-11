"""
Example of using the LLM tool nodes in primeGraph.

This example demonstrates how to create and execute a graph with tool nodes
that interact with LLMs and execute tool functions.
"""

import asyncio
import os
import json
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from primeGraph.graph.llm_clients import LLMClientFactory, Provider
from primeGraph.graph.llm_tools import (
    tool, ToolGraph, ToolEngine, ToolLoopOptions, ToolState, LLMMessage
)


# Example state model with custom fields
class WeatherToolState(ToolState):
    """State for weather-related tool interactions"""
    location: Optional[str] = None
    temperature: Optional[float] = None
    weather_condition: Optional[str] = None
    locations_checked: List[str] = Field(default_factory=list)


# Define some tools using the @tool decorator
@tool("Get current weather for a location")
async def get_weather(location: str) -> Dict:
    """
    Get current weather for a location.
    
    Args:
        location: City or place name
        
    Returns:
        Weather information
    """
    # This would typically call a weather API
    # Simulating API call with mock data
    await asyncio.sleep(0.5)  # Simulate API latency
    
    weather_info = {
        "location": location,
        "temperature": 72.5,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_speed": 5.2,
        "timestamp": time.time()
    }
    
    return weather_info


@tool("Search for information about a topic")
async def search_info(query: str, limit: int = 3) -> List[Dict]:
    """
    Search for information about a topic.
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    # This would typically call a search API
    # Simulating API call with mock data
    await asyncio.sleep(0.7)  # Simulate API latency
    
    results = [
        {
            "title": f"Result 1 for {query}",
            "snippet": f"This is a snippet about {query} with relevant information.",
            "url": f"https://example.com/result1-{query.replace(' ', '-')}"
        },
        {
            "title": f"Result 2 for {query}",
            "snippet": f"Another snippet with information about {query} and related topics.",
            "url": f"https://example.com/result2-{query.replace(' ', '-')}"
        },
        {
            "title": f"Result 3 for {query}",
            "snippet": f"A third information source about {query} with different perspectives.",
            "url": f"https://example.com/result3-{query.replace(' ', '-')}"
        }
    ]
    
    return results[:limit]


@tool("Get current time for a location")
async def get_time(location: str) -> Dict:
    """
    Get current time for a location.
    
    Args:
        location: City or place name
        
    Returns:
        Current time information
    """
    # This would typically call a time API
    # Simulating API call with mock data
    await asyncio.sleep(0.3)  # Simulate API latency
    
    current_time = {
        "location": location,
        "timestamp": time.time(),
        "formatted_time": "3:45 PM",
        "timezone": "UTC-7"
    }
    
    return current_time


async def main():
    # Create a client for the LLM provider
    # Use OpenAI client for this example
    openai_client = LLMClientFactory.create_client(
        Provider.OPENAI, 
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create a graph with our custom state
    graph = ToolGraph("weather_assistant", state_class=WeatherToolState)
    
    # Create tool options
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024,
        stop_on_first_error=True
    )
    
    # Add a tool node with our tools
    tool_node = graph.add_tool_node(
        name="weather_tools",
        tools=[get_weather, search_info, get_time],
        llm_client=openai_client,
        options=options
    )
    
    # Connect the node to the start and end
    graph.add_edge(graph.START, tool_node)
    graph.add_edge(tool_node, graph.END)
    
    # Create the engine
    engine = ToolEngine(graph)
    
    # Set up initial messages in the state
    initial_state = WeatherToolState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful weather assistant. Use the provided tools to answer questions about weather and locations."
        ),
        LLMMessage(
            role="user",
            content="What's the weather like in San Francisco? And what time is it there?"
        )
    ]
    
    # Execute the graph
    print("Running the graph...")
    result = await engine.execute(initial_state=initial_state)
    
    # Access the final state
    final_state = result.state
    
    # Print results
    print("\n=== Final Output ===")
    print(final_state.final_output)
    
    print("\n=== Tool Calls ===")
    for i, tool_call in enumerate(final_state.tool_calls):
        print(f"\nTool Call {i+1}:")
        print(f"  Tool: {tool_call.tool_name}")
        print(f"  Arguments: {json.dumps(tool_call.arguments, indent=2)}")
        print(f"  Result: {tool_call.result}")
        print(f"  Success: {tool_call.success}")
        
    print("\n=== Conversation ===")
    for i, message in enumerate(final_state.messages):
        print(f"\n[{message.role}]:")
        print(message.content)


if __name__ == "__main__":
    asyncio.run(main())