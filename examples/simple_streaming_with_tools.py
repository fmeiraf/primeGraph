#!/usr/bin/env python3
"""
Simple example demonstrating direct use of streaming with Anthropic Claude
and tool calling.

This example shows how to:
1. Stream responses from Claude with tools
2. Handle tool use events in the stream
3. Continue the conversation with tool responses

Usage:
  1. Set your ANTHROPIC_API_KEY environment variable
     export ANTHROPIC_API_KEY=your_api_key_here
  
  2. Run the script:
     python examples/simple_streaming_with_tools.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict

# Make sure primeGraph is importable if running from the examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables directly.")

from primeGraph.graph.llm_clients import (AnthropicClient, StreamingConfig,
                                          StreamingEventType)

# Define tools that Claude can use
WEATHER_TOOL = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature to use"
            }
        },
        "required": ["location"]
    }
}

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": "Perform a mathematical calculation",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}

# Callback function to handle streamed events
def handle_stream_event(event: Dict[str, Any]):
    """Process streamed events and print them appropriately."""
    if event["type"] == "text":
        # Print just the delta text with no newline to simulate streaming
        print("[[starting streaming text event]]")
        print(event["text"], end="", flush=True)
    elif event["type"] == "content_block_stop":
        # Print newlines after content blocks complete
        print("[[streaming content block stop event]]")
    elif event["type"] == "message_stop":
        print("[[streaming message stop event]]")
    elif event["type"] == "tool_use":
        tool_use = event.get("tool_use", {})
        print(f"\n\n[Tool Use Requested]: {tool_use.get('name', 'unknown')}")
        print(f"Tool Input: {json.dumps(tool_use.get('input', {}), indent=2)}\n")

# Function to simulate getting weather data
def get_weather(location: str, unit: str = "celsius") -> str:
    """Simulate a weather API call."""
    temp = 22 if unit == "celsius" else 72
    return json.dumps({
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": "sunny",
        "humidity": 45,
        "forecast": ["sunny", "partly cloudy", "sunny"]
    })

# Function to calculate a mathematical expression
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safely evaluate the expression
        # In a real app, you'd want to use a secure math parser
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

async def main():
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
        # Create Anthropic client directly
        client = AnthropicClient(api_key=api_key)
        
        # Configure streaming
        streaming_config = StreamingConfig(
            enabled=True,
            event_types={
                StreamingEventType.TEXT, 
                StreamingEventType.CONTENT_BLOCK_STOP,
                StreamingEventType.MESSAGE_STOP,
                StreamingEventType.TOOL_USE
            },
            callback=handle_stream_event
        )
        
        # Define tools for Claude
        tools = [WEATHER_TOOL, CALCULATOR_TOOL]
        
        # First user message that will trigger a tool call
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that should use tools to answer questions."
            },
            {
                "role": "user",
                "content": "I'm planning a trip to San Francisco next week. Can you tell me the current weather there? Also, what's 345 * 982?"
            }
        ]
        
        print("\nStarting streaming response with tool calls from Claude...\n")
        print("-" * 50)
        
        # First API call that will trigger tool use
        content, response = await client.generate(
            messages=messages,
            tools=tools,
            model="claude-3-7-sonnet-latest",
            max_tokens=1000,
            streaming_config=streaming_config
        )
        
        # Check if there are tool calls in the response
        tool_calls = []
        if hasattr(response, "content") and isinstance(response.content, list):
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_use = {
                        "id": getattr(block, "id", f"tool_{int(time.time())}"),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {})
                    }
                    tool_calls.append(tool_use)
        
        print(f"response: {response}")
        # If there are tool calls, execute them and continue the conversation
        if tool_calls:
            print("\nExecuting tool calls and continuing the conversation...\n")
            
            # Add the assistant's response with tool calls to messages
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": content}
                ] + [
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]} 
                    for tc in tool_calls
                ]
            })
            
            # Process each tool call and add tool outputs
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                
                # Execute the appropriate tool
                tool_result = ""
                if tool_name == "get_current_weather":
                    location = tool_input.get("location", "unknown")
                    unit = tool_input.get("unit", "celsius")
                    tool_result = get_weather(location, unit)
                elif tool_name == "calculate":
                    expression = tool_input.get("expression", "")
                    tool_result = calculate(expression)
                
                # Add the tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": tool_result
                })
            
            print("\nTools executed, getting final response...\n")
            print("-" * 50)
            
            # Get final response after tool calls
            final_content, final_response = await client.generate(
                messages=messages,
                tools=tools,  # Still include tools in case more tool calls are needed
                model="claude-3-7-sonnet-latest",
                max_tokens=1000,
                streaming_config=streaming_config
            )
        
        print("-" * 50)
        print("\nAll messages in the conversation:")
        
        for idx, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            if role == "user":
                print(f"\n[User #{idx+1}]: {msg.get('content', '')}")
            elif role == "assistant":
                print(f"\n[Assistant #{idx+1}]:")
                content = msg.get('content', '')
                if isinstance(content, list):
                    # Handle composite content with text and tool calls
                    for block in content:
                        if block.get("type") == "text":
                            print(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            print(f"\n[Tool Use: {block.get('name')}]")
                            print(f"Input: {json.dumps(block.get('input', {}), indent=2)}")
                else:
                    print(content)
            elif role == "tool":
                print(f"\n[Tool Response #{idx+1}]: {msg.get('name', '')}")
                print(f"Result: {msg.get('content', '')}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        if "authentication" in str(e).lower():
            print("\nAuthentication failed. Please check your API key.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 