#!/usr/bin/env python3
"""
Example demonstrating the use of streaming with Anthropic Claude in primeGraph.

This example shows how to:
1. Configure streaming options in a tool node
2. Process streamed events using both Redis and callback functions
3. Visualize the streaming response in real-time

Usage:
  1. Set your ANTHROPIC_API_KEY environment variable:
     export ANTHROPIC_API_KEY=your_api_key_here
  
  2. Run with direct callback (simplest):
     python examples/streaming_example.py
     
  3. Run with Redis (requires Docker):
     # Start Redis first
     docker-compose -f docker/docker-compose.yml up redis -d
     # Run Redis consumer in one terminal
     python examples/redis_streaming_consumer.py
     # Run example with Redis in another terminal
     python examples/streaming_example.py --redis
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict

# Make sure primeGraph is importable if running from the examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from primeGraph.constants import END, START
from primeGraph.graph.llm_clients import (LLMClientFactory, Provider,
                                          StreamingConfig, StreamingEventType)
from primeGraph.graph.llm_tools import (ToolGraph, ToolLoopOptions, ToolState,
                                        tool)


# Define a simple tool
@tool("Get the current time")
async def get_time(**kwargs):
    """Get the current time in ISO format."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

@tool("Echo the input text")
async def echo_text(text: str, **kwargs):
    """Echo back the input text."""
    return f"You said: {text}"

# Callback function to handle streamed events
def handle_stream_event(event: Dict[str, Any]):
    if event["type"] == "text":
        # Print just the delta text with no newline to simulate streaming
        print(event["text"], end="", flush=True)
    elif event["type"] == "content_block_stop":
        # Print newlines after content blocks complete
        print("\n")
    elif event["type"] == "message_stop":
        print("\n[Message complete]\n")

async def main(use_redis=False, redis_host="localhost", redis_port=6379):
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
    
    try:
        # Create Anthropic client
        llm_client = LLMClientFactory.create_client(Provider.ANTHROPIC, api_key=api_key)
        
        # Create a tool graph with one node
        graph = ToolGraph(
            name="Streaming Example",
            state=ToolState(),
        )
        
        # Configure the channel name for streaming
        channel_name = f"llm_stream_{int(time.time())}"
        print(f"Using channel name: {channel_name}")
        
        # Set up streaming options with either Redis or callback
        if use_redis:
            print(f"Using Redis at {redis_host}:{redis_port} with channel {channel_name}")
            print("Make sure to run redis_streaming_consumer.py in another terminal with:")
            print(f"  python examples/redis_streaming_consumer.py --channel {channel_name} --host {redis_host} --port {redis_port}")
            
            streaming_config = StreamingConfig(
                enabled=True,
                event_types={
                    StreamingEventType.TEXT, 
                    StreamingEventType.CONTENT_BLOCK_STOP,
                    StreamingEventType.MESSAGE_STOP,
                    StreamingEventType.TOOL_USE
                },
                redis_host=redis_host,
                redis_port=redis_port,
                redis_channel=channel_name
            )
        else:
            print("Using direct callback for streaming (no Redis)")
            streaming_config = StreamingConfig(
                enabled=True,
                event_types={
                    StreamingEventType.TEXT, 
                    StreamingEventType.CONTENT_BLOCK_STOP,
                    StreamingEventType.MESSAGE_STOP
                },
                callback=handle_stream_event
            )
        
        # Create tool options with streaming enabled
        options = ToolLoopOptions(
            max_iterations=1,
            model="claude-3-7-sonnet-latest",
            max_tokens=1000,
            streaming_config=streaming_config
        )
        
        # Add a tool node to the graph
        tools = [get_time, echo_text]
        node = graph.add_tool_node(
            name="streaming_node",
            tools=tools,
            llm_client=llm_client,
            options=options
        )
        
        # Connect the node to start and end
        graph.add_edge(START, "streaming_node")
        graph.add_edge("streaming_node", END)
        
        # Initialize the state with a prompt
        graph.state.messages.append({
            "role": "user",
            "content": "Please write a creative short story about a programmer who discovers an AI that can predict the future. The story should be about 3 paragraphs long."
        })
        
        # Give the Redis consumer time to set up if using Redis
        if use_redis:
            print("\nWaiting 2 seconds for Redis consumer to be ready...")
            await asyncio.sleep(2)
        
        # Execute the graph
        print("\nStarting LLM streaming call...\n")
        print("-" * 50)
        await graph.execute()
        print("-" * 50)
        
        # Show final output from state
        if hasattr(graph.state, "final_output") and graph.state.final_output:
            print(f"\nFinal output: {graph.state.final_output}")
        
        # Alternative way to use streaming with just the options flags:
        print("\nAlternative way using option flags:\n")
        graph2 = ToolGraph(
            name="Streaming Example 2",
            state=ToolState(),
        )
        
        # Use streaming option flags directly
        options2 = ToolLoopOptions(
            max_iterations=1,
            model="claude-3-7-sonnet-latest",
            max_tokens=500,
            stream=True,  # Enable streaming
            stream_events={StreamingEventType.TEXT},  # Only stream text events
            stream_callback=handle_stream_event if not use_redis else None,  # Set callback if not using Redis
            redis_host=redis_host if use_redis else None,
            redis_port=redis_port if use_redis else None,
            redis_channel=f"{channel_name}_2" if use_redis else None
        )
        
        if use_redis:
            print(f"Second example using Redis channel: {channel_name}_2")
        
        # Add tool node and connect
        node2 = graph2.add_tool_node(
            name="streaming_node",
            tools=tools,
            llm_client=llm_client,
            options=options2
        )
        
        graph2.add_edge(START, "streaming_node")
        graph2.add_edge("streaming_node", END)
        
        # Initialize with a different prompt
        graph2.state.messages.append({
            "role": "user",
            "content": "Write a brief haiku about programming."
        })
        
        # Execute the second graph
        print("\nStarting second LLM streaming call...\n")
        print("-" * 50)
        await graph2.execute()
        print("-" * 50)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        if "authentication" in str(e).lower():
            print("\nAuthentication failed. Please check your API key.")
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="primeGraph Streaming Example")
    parser.add_argument("--redis", action="store_true", help="Use Redis for streaming")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    
    args = parser.parse_args()
    
    sys.exit(asyncio.run(main(
        use_redis=args.redis,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    ))) 