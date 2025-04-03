#!/usr/bin/env python3
"""
Example demonstrating how to use ToolGraph with Redis streaming support and tool calls.

This example shows how to:
1. Define tools using the @tool decorator
2. Create a ToolGraph with tool nodes
3. Configure Redis streaming in ToolLoopOptions
4. Execute the graph with Redis-based streaming of responses

For this example to work, you need to:
1. Have Redis running (using Docker or locally)
2. Run the redis_streaming_consumer.py script in another terminal to see the stream

Usage:
  1. Set your environment variables (or use .env file):
     export ANTHROPIC_API_KEY=your_api_key_here
     export REDIS_HOST=localhost
     export REDIS_PORT=6379
     export REDIS_CHANNEL=your_custom_channel  # Optional
     export USER_ID=your_user_id              # Optional
     export SESSION_ID=your_session_id        # Optional
  
  2. Start Redis if not already running:
     docker-compose up -d redis
     
  3. In a separate terminal, run the Redis consumer:
     python examples/redis_streaming_consumer.py
  
  4. Run this script:
     python examples/tool_graph_redis_streaming.py
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict

from pydantic import Field
from rich import print as rprint

from primeGraph import END, START
from primeGraph.buffer.factory import LastValue

# Make sure primeGraph is importable if running from the examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables directly.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Note: Redis package not installed. Install with 'pip install redis'")
    print("Redis streaming will not be available without this package.")

from primeGraph.graph.llm_clients import (AnthropicClient, StreamingConfig,
                                          StreamingEventType)
from primeGraph.graph.llm_tools import (LLMMessage, ToolGraph, ToolLoopOptions,
                                        ToolState, ToolType, tool)


# Define a custom state class to store conversation context
class WeatherToolState(ToolState):
    user_location: LastValue[str] = Field(default="")
    temperature: LastValue[float] = Field(default=0)
    condition: LastValue[str] = Field(default="")
    calculation_result: LastValue[float] = Field(default=0)
    user_id: LastValue[str] = Field(default="demo_user")
    session_id: LastValue[str] = Field(default="demo_session")

# Define tools using the @tool decorator
@tool("Get the current weather in a given location", tool_type=ToolType.FUNCTION)
async def get_current_weather(location: str, unit: str = "celsius", state: WeatherToolState = None) -> Dict[str, Any]:
    """Simulate getting weather data for a location."""
    print(f"\n[TOOL] Getting weather for {location}...")
    
    # Simulate a weather API call
    temp = 22 if unit == "celsius" else 72
    condition = "sunny"
    
    # Store in state
    if state:
        state.user_location = location
        state.temperature = temp
        state.condition = condition
    
    return {
        "weather": {
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": condition,
            "humidity": 45,
            "forecast": ["sunny", "partly cloudy", "sunny"]
        }
    }

@tool("Perform a mathematical calculation", tool_type=ToolType.FUNCTION)
async def calculate(expression: str, state: WeatherToolState = None) -> Dict[str, Any]:
    """Safely evaluate a mathematical expression."""
    print(f"\n[TOOL] Calculating: {expression}...")
    
    try:
        # Safely evaluate the expression (in a real app, use a secure math parser)
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
        
        # Store in state
        if state:
            state.calculation_result = result
        
        return {
            "calculation": {
                "expression": expression,
                "result": result
            }
        }
    except Exception as e:
        return {
            "calculation": {
                "expression": expression,
                "error": str(e)
            }
        }

async def main():
    # Get configuration from environment variables with fallbacks
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    redis_channel = os.environ.get("REDIS_CHANNEL", "llm_stream")
    user_id = os.environ.get("USER_ID", "demo_user")
    session_id = os.environ.get("SESSION_ID", "demo_session")
    
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key with:")
        print("export ANTHROPIC_API_KEY=your_api_key_here")
        return 1
    
    if api_key.lower().startswith("your_api_key"):
        print("ERROR: Please replace the placeholder with your actual Anthropic API key")
        print("export ANTHROPIC_API_KEY=your_actual_key_here")
        return 1
    
    # Check if Redis is available
    if not REDIS_AVAILABLE:
        print("ERROR: Redis package is required for this example")
        print("Install it with: pip install redis")
        return 1
    
    try:
        # Create Anthropic client
        llm_client = AnthropicClient(api_key=api_key)
        
        # Initialize state with user and session IDs
        state = WeatherToolState()
        state.user_id = user_id
        state.session_id = session_id
        
        # Create a Redis-specific channel using user and session IDs
        channel = f"user:{state.user_id}:session:{state.session_id}"
        if redis_channel != "llm_stream":
            channel = redis_channel
        
        # Display configuration
        print("\nConfiguration:")
        print(f"Redis Host: {redis_host}")
        print(f"Redis Port: {redis_port}")
        print(f"Redis Channel: {channel}")
        print(f"User ID: {user_id}")
        print(f"Session ID: {session_id}")
        
        # Configure streaming with Redis
        streaming_config = StreamingConfig(
            enabled=True,
            event_types={
                StreamingEventType.MESSAGE_START,
                StreamingEventType.TEXT, 
                StreamingEventType.CONTENT_BLOCK_STOP,
                StreamingEventType.MESSAGE_STOP,
                StreamingEventType.TOOL_USE
            },
            redis_host=redis_host,
            redis_port=redis_port,
            redis_channel=channel,
            event_type_mapping={
                StreamingEventType.TEXT: "CUSTOM_TEXT",
                StreamingEventType.TOOL_USE: "CUSTOM_TOOL",
                StreamingEventType.MESSAGE_STOP: "CUSTOM_STOP"
            }
        )
        
        # Create a tool loop options object with streaming configuration
        tool_options = ToolLoopOptions(
            model="claude-3-7-sonnet-latest",
            max_tokens=8192,
            streaming_config=streaming_config,
            max_iterations=5  # Limit the number of tool calls for this example
        )
        
        # Create a tool graph
        graph = ToolGraph(
            name="WeatherAndCalculatorGraph",
            state=state,
            max_iterations=5,
            verbose=True
        )
        
        # Add a tool node with both tools
        tools = [get_current_weather, calculate]
        tool_node = graph.add_tool_node(
            name="weather_and_calculator_node",
            tools=tools,
            llm_client=llm_client,
            options=tool_options
        )

        graph.add_edge(START, tool_node.name)
        graph.add_edge(tool_node.name, END)
        
        # Set up the initial user message using the LLMMessage class instead of dictionaries
        initial_message = "I'm planning a trip to San Francisco next week. Can you tell me the current weather there? Also, what's 345 * 982?"
        
        # Use LLMMessage objects instead of dictionaries
        state.messages.append(LLMMessage(
            role="system",
            content="You are a helpful assistant that uses tools to answer questions accurately."
        ))
        
        state.messages.append(LLMMessage(
            role="user",
            content=initial_message
        ))
        
        print("\nStarting tool graph execution with Redis streaming...\n")
        print("-" * 60)
        print(f"User: {initial_message}")
        print("-" * 60)
        print(f"Streaming events to Redis channel: {channel}")
        print("Make sure redis_streaming_consumer.py is running in another terminal")
        print(f"Run: python examples/redis_streaming_consumer.py --channel {channel}")
        print("-" * 60)
        
        # Execute the graph
        chain_id = await graph.execute()
        
        print("-" * 60)
        print("\nTool graph execution complete.")
        print(f"Chain ID: {chain_id}")

        rprint(state)
        
        # Display final state information
        if state.user_location:
            print(f"\nWeather information for {state.user_location}:")
            print(f"Temperature: {state.temperature}°C")
            print(f"Condition: {state.condition}")
        
        if state.calculation_result:
            print(f"\nCalculation result: {state.calculation_result}")
        
        # Print all messages in the conversation
        print("\nFull conversation:")
        for idx, msg in enumerate(state.messages):
            role = msg.role
            if role == "user":
                print(f"\n[User #{idx+1}]: {msg.content}")
            elif role == "assistant":
                print(f"\n[Assistant #{idx+1}]:")
                content = msg.content
                if isinstance(content, list):
                    # Handle composite content with text and tool calls
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                print(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                print(f"\n[Tool Use: {block.get('name')}]")
                                print(f"Input: {json.dumps(block.get('input', {}), indent=2)}")
                        else:
                            print(str(block))
                else:
                    print(content)
            elif role == "tool":
                print(f"\n[Tool Response #{idx+1}]: {msg.name if hasattr(msg, 'name') else ''}")
                print(f"Result: {msg.content}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 