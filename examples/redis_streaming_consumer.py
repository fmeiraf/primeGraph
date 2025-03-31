#!/usr/bin/env python3
"""
Redis consumer example for primeGraph streaming.

This script listens to Redis channels where streaming events are published
and displays them in real-time. It can be used with the streaming_example.py
script by enabling the Redis configuration in that example.

Run this script in a separate terminal window before running the streaming example
with Redis enabled.
"""

import argparse
import asyncio
import json
import os
import sys

# Make sure primeGraph is importable if running from the examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import redis
except ImportError:
    print("Redis package not installed. Install with 'pip install redis'")
    sys.exit(1)

def print_text_event(event):
    """Print text events without newlines to simulate streaming."""
    text = event.get("text", "")
    print(text, end="", flush=True)

def print_content_block_stop(event):
    """Print content block stop events."""
    print("\n\n[Content block complete]")
    content_block = event.get("content_block", {})
    if content_block and content_block.get("type") == "text":
        print(f"Total length: {len(content_block.get('text', ''))}")

def print_message_stop(event):
    """Print message stop events."""
    print("\n\n[Message complete]")
    message = event.get("message", {})
    if message:
        print(f"Message ID: {message.get('id')}")

def print_tool_use(event):
    """Print tool use events."""
    print("\n\n[Tool use]")
    tool_use = event.get("tool_use", {})
    if tool_use:
        print(f"Tool: {tool_use.get('name')}")
        print(f"Input: {tool_use.get('input')}")

EVENT_HANDLERS = {
    "text": print_text_event,
    "content_block_stop": print_content_block_stop,
    "message_stop": print_message_stop,
    "tool_use": print_tool_use,
    "input_json": lambda e: print(f"\n[JSON] {e.get('partial_json', '')}")
}

def process_event(event_data):
    """Process a streaming event from Redis."""
    try:
        event = json.loads(event_data)
        event_type = event.get("type")
        
        if event_type in EVENT_HANDLERS:
            EVENT_HANDLERS[event_type](event)
        else:
            print(f"\n[Unknown event type: {event_type}]")
    except json.JSONDecodeError:
        print(f"\n[Error decoding JSON: {event_data}]")
    except Exception as e:
        print(f"\n[Error processing event: {str(e)}]")

async def listen_to_redis_channel(channel, host="localhost", port=6379):
    """Listen to Redis channel and process streaming events."""
    print(f"Connecting to Redis at {host}:{port}")
    print(f"Listening for streaming events on channel: {channel}")
    print("Waiting for events... (Press Ctrl+C to exit)\n")
    
    r = redis.Redis(host=host, port=port)
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    
    try:
        # Process messages in a loop
        for message in pubsub.listen():
            if message["type"] == "message":
                process_event(message["data"])
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pubsub.unsubscribe()
        r.close()

def main():
    """Parse arguments and run Redis listener."""
    parser = argparse.ArgumentParser(description="Listen for primeGraph streaming events on Redis")
    parser.add_argument(
        "--channel", 
        default="llm_stream", 
        help="Redis channel to listen on (default: llm_stream)"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6379, 
        help="Redis port (default: 6379)"
    )
    
    args = parser.parse_args()
    
    try:
        # Use asyncio to handle the Redis listener
        asyncio.run(listen_to_redis_channel(args.channel, args.host, args.port))
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 