#!/usr/bin/env python3
"""
Simple example demonstrating direct use of streaming with Anthropic Claude
without using the primeGraph tool system.

This is useful if you just want to stream Claude responses directly.

Usage:
  1. Set your ANTHROPIC_API_KEY environment variable
     export ANTHROPIC_API_KEY=your_api_key_here
  
  2. Run the script:
     python examples/simple_streaming.py
"""

import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

from primeGraph.graph.llm_clients import (AnthropicClient, StreamingConfig,
                                          StreamingEventType)

load_dotenv()

# Make sure primeGraph is importable if running from the examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




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
                StreamingEventType.MESSAGE_STOP
            },
            callback=handle_stream_event  # Add the callback function
        )
        
        # Define a simple conversation
        messages = [
            {
                "role": "user",
                "content": "Please write a haiku about streaming LLM responses."
            }
        ]
        
        print("\nStarting streaming response from Claude...\n")
        print("-" * 50)
        
        # Call the API with streaming enabled
        content, _ = await client.generate(
            messages=messages,
            model="claude-3-7-sonnet-latest",
            max_tokens=300,
            streaming_config=streaming_config
        )
        
        print("-" * 50)
        print("\nFinal content from response object:")
        print(content)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        if "authentication" in str(e).lower():
            print("\nAuthentication failed. Please check your API key.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 