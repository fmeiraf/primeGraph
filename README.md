<p align="center">
  <img src="docs/images/logo_art.png" alt="primeGraph Logo" width="200"/>
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Package Version](https://img.shields.io/badge/package-1.9.10-blue.svg)](https://pypi.org/project/primegraph/)

---

## Overview

**primeGraph** is a Python library for building and executing workflows using graphs, ranging from simple sequential processes to complex parallel execution patterns. While originally optimized for AI applications, its flexible architecture makes it suitable for any workflow orchestration needs.

Key principles:

- **Flexibility First**: Design your nodes and execution patterns with complete freedom.
- **Zero Lock-in**: Deploy and run workflows however you want, with no vendor dependencies.
- **Opinionated Yet Adaptable**: Structured foundations with room for customization.

_Note from the author: This project came to life through my experience creating AI applications. I want to acknowledge [langgraph](https://www.langchain.com/langgraph) as the main inspiration for this project. As an individual developer, I wanted to gain experience creating my own workflow engine to implement more of my own ideas and learnings. At the same time, I also wanted to create a framework that is flexible enough for others to deploy their apps however they want, as this is an open source project. So feel free to use it, modify it, and contribute to it._

#### Features

- **Flexible Graph Construction**: Build multiple workflows with sequential and parallel execution paths.
- **State Management**: Built-in state management with various buffer types to coordinate state updates during execution.
- **Type Safety**: Uses Pydantic for enforcing shared state types across nodes.
- **Router Nodes**: Dynamically choose execution paths based on node outputs.
- **Repeatable Nodes**: Execute nodes multiple times either in parallel or sequence.
- **Subgraphs**: Compose graphs of subgraphs to design complex workflows.
- **LLM Tool Integration**: Specialized nodes for LLM interaction with tool/function calling capabilities.
- **Tool Pause/Resume**: Ability to pause execution before or after specific tool calls for human review and approval.
- **Persistence**: Save and resume execution using checkpoint storage (supports memory and Postgres).
- **Async Support**: Uses async/await for non-blocking execution with engine methods `execute()` and `resume()`.
- **Flow Control**: Supports human-in-the-loop interactions by pausing and resuming workflows.
- **Visualization**: Generate visual representations of your workflows with minimal effort.
- **Web Integration**: Integrate with FastAPI and WebSockets for interactive workflows.
- **Streaming Support**: Stream LLM outputs in real-time with Redis-based event streaming.

## Installation

```bash
pip install primeGraph
```

#### [Optional] Install Graphviz for visualization

To have the `graph.visualize()` method work, install the Graphviz binary:

https://graphviz.org/download/

## Core Features

### The Basics

```python
import asyncio
from primeGraph import Graph, START, END
from primeGraph.models import GraphState
from primeGraph.buffer import History, LastValue, Incremental

# Define your state with appropriate buffer types
class DocumentProcessingState(GraphState):
    processed_files: History[str]      # Stores all returned file names
    current_status: LastValue[str]       # Keeps the last status value
    number_of_executed_steps: Incremental[int]  # Increments with each step

# Initialize state
state = DocumentProcessingState(
    processed_files=[],
    current_status="initializing",
    number_of_executed_steps=0
)

# Create a graph with the state
graph = Graph(state=state)

# Adding nodes to the graph
'to simulate the workflow
@graph.node()
def load_documents(state):
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

@graph.node()
def validate_documents(state):
    return {
        "current_status": "validating",
        "number_of_executed_steps": 1
    }

@graph.node()
def process_documents(state):
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

# Connect nodes
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "validate_documents")
graph.add_edge("validate_documents", "process_documents")
graph.add_edge("process_documents", END)

# Compile the graph
graph.compile()

# Execute the graph asynchronously using the new engine methods
async def run_graph():
    await graph.execute()
    print(state)
    graph.visualize()

asyncio.run(run_graph())
```

<p align="center">
  <img src="docs/images/readme_base_usage.png" alt="Basic Usage Graph Visualization" width="400"/>
</p>

### Router Nodes

```python
import asyncio
from primeGraph import Graph, START, END

graph = Graph()

@graph.node()
def load_documents(state):
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

@graph.node()
def validate_documents(state):
    return {
        "current_status": "validating",
        "number_of_executed_steps": 1
    }

@graph.node()
def process_documents(state):
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

@graph.node()
def route_documents(state):
    if "invoice" in state.current_status:
        return "process_invoice"
    return "cancel_invoice"

@graph.node()
def process_invoice(state):
    return {"current_status": "invoice_processed"}

@graph.node()
def cancel_invoice(state):
    return {"current_status": "invoice_cancelled"}

# Connect nodes and define router edges
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "validate_documents")
graph.add_edge("validate_documents", "process_documents")

# Router node is connected as edge from process_documents
graph.add_router_edge("process_documents", "route_documents")
graph.add_edge("process_invoice", END)
graph.add_edge("cancel_invoice", END)

# Compile and execute
graph.compile()

async def run_router():
    await graph.execute()
    print(state)
    graph.visualize()

import asyncio
asyncio.run(run_router())
```

<p align="center">
  <img src="docs/images/readme_router_nodes.png" alt="Router Nodes visualization" width="400"/>
</p>

### Repeatable Nodes

```python
import asyncio
from primeGraph import Graph, START, END

graph = Graph()

@graph.node()
def load_documents(state):
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

@graph.node()
def validate_documents(state):
    return {
        "current_status": "validating",
        "number_of_executed_steps": 1
    }

@graph.node()
def process_documents(state):
    return {
        "current_status": "processing",
        "number_of_executed_steps": 1
    }

@graph.node()
def repeating_process_batch(state):
    return {
        "processed_files": f"batch_{state.number_of_executed_steps}",
        "number_of_executed_steps": 1
    }

@graph.node()
def conclude_documents(state):
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

# Connect nodes
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "validate_documents")
graph.add_edge("validate_documents", "process_documents")

# Add repeating edge for processing multiple batches
graph.add_repeating_edge(
    "process_documents",
    "repeating_process_batch",
    "conclude_documents",
    repeat=3,
    parallel=True
)
graph.add_edge("conclude_documents", END)

graph.compile()

async def run_repeatable():
    await graph.execute()
    print(state)
    graph.visualize()

import asyncio
asyncio.run(run_repeatable())
```

<p align="center">
  <img src="docs/images/readme_repeatable_nodes.png" alt="Repeatable Nodes visualization" width="400"/>
</p>

### Subgraphs

```python
import asyncio
from primeGraph import Graph, START, END

state = ... # initialize your state appropriately
main_graph = Graph(state=state)

@main_graph.node()
def load_documents(state):
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

@main_graph.subgraph()
def validation_subgraph():
    subgraph = Graph(state=state)

    @subgraph.node()
    def check_format(state):
        return {"current_status": "checking_format"}

    @subgraph.node()
    def verify_content(state):
        return {"current_status": "verifying_content"}

    subgraph.add_edge(START, "check_format")
    subgraph.add_edge("check_format", "verify_content")
    subgraph.add_edge("verify_content", END)

    return subgraph

@main_graph.node()
def pre_process_documents(state):
    return {
        "current_status": "pre_processed",
        "number_of_executed_steps": 1
    }

@main_graph.node()
def conclude_documents(state):
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

# Connect nodes
main_graph.add_edge(START, "load_documents")
main_graph.add_edge("load_documents", "validation_subgraph")
main_graph.add_edge("load_documents", "pre_process_documents")
main_graph.add_edge("validation_subgraph", "conclude_documents")
main_graph.add_edge("pre_process_documents", "conclude_documents")
main_graph.add_edge("conclude_documents", END)

main_graph.compile()

async def run_subgraph():
    await main_graph.execute()
    print(state)
    main_graph.visualize()

import asyncio
asyncio.run(run_subgraph())
```

<p align="center">
  <img src="docs/images/readme_subgraphs.png" alt="Subgraphs visualization" width="400"/>
</p>

### Flow Control

```python
import asyncio
from primeGraph import Graph, START, END

state = ...  # initialize state

graph = Graph(state=state)

@graph.node()
def load_documents(state):
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

# This node will interrupt execution before running
@graph.node(interrupt="before")
def review_documents(state):
    return {
        "current_status": "validating",
        "number_of_executed_steps": 1
    }

@graph.node()
def process_documents(state):
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

# Connect nodes
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "review_documents")
graph.add_edge("review_documents", "process_documents")
graph.add_edge("process_documents", END)

graph.compile()

async def run_flow():
    # Start execution; this will pause before 'review_documents'
    await graph.execute()
    print("State after interruption:", state)

    # Once ready, resume the execution
    await graph.resume()
    print("State after resuming:", state)
    graph.visualize()

import asyncio
asyncio.run(run_flow())
```

<p align="center">
  <img src="docs/images/readme_interrupt.png" alt="Flow Control visualization" width="400"/>
</p>

#### Persistence

```python
from primeGraph.checkpoint.postgresql import PostgreSQLStorage

# Configure storage
storage = PostgreSQLStorage.from_config(
    host="localhost",
    database="documents_db",
    user="user",
    password="password"
)

# Create graph with checkpoint storage
graph = Graph(state=state, checkpoint_storage=storage)

@graph.node(interrupt="before")
def validate_documents(state):
    return {"current_status": "needs_review"}

# Execute graph and save checkpoint
async def run_with_checkpoint():
    chain_id = await graph.execute()
    # Later, load and resume from checkpoint
    graph.load_from_checkpoint(chain_id)
    await graph.resume()

import asyncio
asyncio.run(run_with_checkpoint())
```

#### Async Support

```python
import asyncio

@graph.node()
async def async_document_process(state):
    await asyncio.sleep(1)  # Simulate async processing
    return {
        "processed_files": "async_processed",
        "current_status": "async_complete"
    }

# Execute the async graph
async def run_async():
    await graph.execute()
    # Resume if execution was paused
    await graph.resume()

asyncio.run(run_async())
```

#### Web Integration

```python
import os
import logging
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from primeGraph.buffer import History
from primeGraph.checkpoint import LocalStorage
from primeGraph import Graph, START, END
from primeGraph.models import GraphState
from primeGraph.web import create_graph_service, wrap_graph_with_websocket

logging.basicConfig(level=logging.DEBUG)

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
async def hello():
    return {"message": "Hello World"}

graphs: List[Graph] = []

# Define a simple state model
class SimpleGraphState(GraphState):
    messages: History[str]

state = SimpleGraphState(messages=[])

storage = LocalStorage()
graph1 = Graph(state=state, checkpoint_storage=storage)

@graph1.node()
def add_hello(state: GraphState):
    logging.debug("add_hello")
    return {"messages": "Hello"}

@graph1.node()
def add_world(state: GraphState):
    logging.debug("add_world")
    return {"messages": "World"}

@graph1.node()
def add_exclamation(state: GraphState):
    logging.debug("add_exclamation")
    return {"messages": "!"}

graph1.add_edge(START, "add_hello")
graph1.add_edge("add_hello", "add_world")
graph1.add_edge("add_world", "add_exclamation")
graph1.add_edge("add_exclamation", END)

graph1.compile()

service = create_graph_service(graph1, storage, path_prefix="/graphs/workflow1")
app.include_router(service.router, tags=["workflow1"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Basic Usage Examples

### Chatbot Example

```python
import asyncio
from primeGraph import Graph, START, END
from primeGraph.models import GraphState
from primeGraph.buffer import History, LastValue
from pydantic import BaseModel, Field
import logging

class ChatbotState(GraphState):
    chat_history: History[dict[str, str]]
    user_wants_to_exit: LastValue[bool] = Field(default=False)

class ChatbotResponse(BaseModel):
    chat_message: str
    user_requested_to_quit: bool = Field(description="True if user wants to quit")

chatbot_state = ChatbotState(chat_history=[], user_wants_to_exit=False)
chatbot_graph = Graph(state=chatbot_state, verbose=False)

@chatbot_graph.node(interrupt="before")
def chat_with_user(state):
    # Simulate calling an AI service
    try:
        # Replace with actual call
        response = ChatbotResponse(chat_message="Hello, how can I assist?", user_requested_to_quit=False)
        print(response.chat_message)
        return {"chat_history": {"role": "assistant", "content": response.chat_message},
                "user_wants_to_exit": response.user_requested_to_quit}
    except Exception as e:
        raise e

@chatbot_graph.node()
def assess_next_step(state):
    if state.user_wants_to_exit:
        return END
    return "chat_with_user"

chatbot_graph.add_edge(START, "chat_with_user")
chatbot_graph.add_router_edge("chat_with_user", "assess_next_step")
chatbot_graph.compile()
chatbot_graph.visualize()

async def run_chatbot():
    await chatbot_graph.execute()

    def add_user_message(message: str):
        chatbot_graph.update_state_and_checkpoint({"chat_history": {"role": "user", "content": message}})

    while not chatbot_state.user_wants_to_exit:
        user_input = input("Your message: ")
        print(f"You: {user_input}")
        add_user_message(user_input)
        await chatbot_graph.resume()

    print("Bye")

import asyncio
asyncio.run(run_chatbot())
```

### Async Workflow

```python
import asyncio
from primeGraph import Graph, START, END
from primeGraph.models import GraphState
from primeGraph.buffer import History, LastValue
from pydantic import BaseModel

# Define models for async workflow
class Character(GraphState):
    character_name: LastValue[str]
    character_items: History[tuple[str,str]]
    character_summary: LastValue[str]

class CharacterName(BaseModel):
    character_name: str

class CharacterSummary(BaseModel):
    character_summary: str

class CharacterItem(BaseModel):
    item_name: str
    item_description: str

character_state = Character(character_name="", character_items=[], character_summary="")
character_graph = Graph(state=character_state, verbose=False)

@character_graph.node()
async def pick_character_name(state):
    # Simulate an async call
    return {"character_name": "Frodo Baggins"}

@character_graph.node()
async def pick_character_profession(state):
    return {"character_items": ("Adventurer", "Embarks on quests")}

@character_graph.node()
async def pick_character_apparel(state):
    return {"character_items": ("Mystic Robe", "Adorned with ancient runes")}

@character_graph.node()
async def pick_character_partner(state):
    return {"character_items": ("Samwise Gamgee", "Loyal companion")}

@character_graph.node()
async def create_charater_summary(state):
    ch_items = "\n".join([f"{item[0]}: {item[1]}" for item in state.character_items])
    return {"character_summary": f"{state.character_name} is accompanied by:\n{ch_items}"}

character_graph.add_edge(START, "pick_character_name")
character_graph.add_edge("pick_character_name", "pick_character_profession")
character_graph.add_edge("pick_character_name", "pick_character_apparel")
character_graph.add_edge("pick_character_name", "pick_character_partner")
character_graph.add_edge("pick_character_profession", "create_charater_summary")
character_graph.add_edge("pick_character_apparel", "create_charater_summary")
character_graph.add_edge("pick_character_partner", "create_charater_summary")
character_graph.add_edge("create_charater_summary", END)

character_graph.compile()

async def run_async_workflow():
    await character_graph.execute()
    print(character_graph.state)

import asyncio
asyncio.run(run_async_workflow())
```

---

Note: All examples now use the new asynchronous engine methods `execute()` and `resume()`. For scripts, wrap these calls with `asyncio.run(...)` or use an async context as needed.

### LLM Tool Nodes

```python
import asyncio
from typing import Dict, List, Any
from primeGraph import START, END
from primeGraph.graph.llm_tools import (
    tool, ToolNode, ToolGraph, ToolState,
    ToolLoopOptions, LLMMessage
)
from primeGraph.graph.llm_clients import OpenAIClient, AnthropicClient

# Define state for your tool-based workflow
class ResearchState(ToolState):
    search_results: List[Dict[str, Any]] = []
    final_summary: str = None

# Define tools with the @tool decorator
@tool("Search the web for information")
async def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Search the web for information on a topic"""
    # Implementation would call a real search API
    return {
        "results": [
            {"title": "Example result 1", "content": "Example content..."},
            {"title": "Example result 2", "content": "More example content..."}
        ]
    }

@tool("Summarize information")
async def summarize(text: str) -> Dict[str, Any]:
    """Summarize text into a concise summary"""
    # Implementation would use an LLM or summarization service
    return {"summary": f"Summarized version of: {text[:30]}..."}

# Create a tool with state parameter for direct state manipulation
@tool("Store search information in state")
async def save_to_state(result_id: str, state=None) -> Dict[str, Any]:
    """Save search results directly to state"""
    if state and hasattr(state, "search_results"):
        state.search_results.append({"id": result_id, "saved_at": "now"})
    return {"status": "saved", "result_id": result_id}

# Create a graph for tool-based workflow
graph = ToolGraph("research_workflow", state_class=ResearchState)

# Initialize an LLM client (OpenAI or Anthropic)
llm_client = OpenAIClient(api_key="your-api-key-here")

# Add a tool node to the graph
node = graph.add_tool_node(
    name="researcher",
    tools=[search_web, summarize, save_to_state],
    llm_client=llm_client,
    options=ToolLoopOptions(max_iterations=5)
)

# Connect nodes
graph.add_edge(START, node.name)
graph.add_edge(node.name, END)

# Execute the graph
async def run_research():
    # Create initial state with user query
    initial_state = ResearchState()
    initial_state.messages = [
        LLMMessage(role="system", content="You are a helpful research assistant."),
        LLMMessage(role="user", content="Research quantum computing advancements in 2023")
    ]

    # Execute the graph
    await graph.execute(initial_state=initial_state)

    # Access final state
    final_state = graph.state
    print(f"Tool calls: {len(final_state.tool_calls)}")
    print(f"Final output: {final_state.final_output}")
    print(f"Saved search results: {final_state.search_results}")

asyncio.run(run_research())
```

### LLM Message Callbacks

You can register callbacks to receive messages from LLMs during tool execution:

```python
# Define callback functions for LLM messages and tool usage
def on_message(message_data):
    """Callback triggered whenever the LLM generates a message"""
    print(f"LLM MESSAGE: {message_data['content']}")
    print(f"Is final: {message_data['is_final']}")
    if message_data['has_tool_calls']:
        print(f"Contains {len(message_data['tool_calls'])} tool calls")

def on_tool_use(tool_data):
    """Callback triggered when a tool is executed"""
    print(f"TOOL EXECUTED: {tool_data['name']}")
    print(f"Arguments: {tool_data['arguments']}")
    print(f"Result: {tool_data['result']}")

# Register callbacks when creating the tool node
node = graph.add_tool_node(
    name="researcher",
    tools=[search_web, summarize],
    llm_client=llm_client,
    options=ToolLoopOptions(max_iterations=5),
    on_message=on_message,  # Register message callback
    on_tool_use=on_tool_use  # Register tool use callback
)
```

### Tool Pause and Resume

```python
# Define tools that pause before or after execution
@tool("Process payment", pause_before_execution=True)
async def process_payment(order_id: str, amount: float) -> Dict[str, Any]:
    """Process a payment for an order, pausing before execution for input verification"""
    return {
        "order_id": order_id,
        "amount": amount,
        "status": "processed",
        "transaction_id": f"TX-{order_id}-{int(time.time())}"
    }

@tool("Update account", pause_after_execution=True)
async def update_account(user_id: str, new_email: str) -> Dict[str, Any]:
    """Update a user account, pausing after execution for result verification"""
    return {
        "user_id": user_id,
        "email": new_email,
        "status": "updated",
        "timestamp": int(time.time())
    }

# Add to graph
tool_node = graph.add_tool_node(
    name="account_processor",
    tools=[process_payment, update_account, get_user_details],
    llm_client=llm_client
)

# Execute the graph - will pause before/after tool execution depending on configuration
await graph.execute(initial_state)

# Check if paused before execution
if graph.state.is_paused and not graph.state.paused_after_execution:
    print("Paused before execution")
    print(f"Tool: {graph.state.paused_tool_name}")
    print(f"Arguments: {graph.state.paused_tool_arguments}")

    # Resume with execution (approve) or skip (reject)
    await graph.resume(execute_tool=True)  # approve
    # await graph.resume(execute_tool=False)  # reject

# Check if paused after execution
elif graph.state.is_paused and graph.state.paused_after_execution:
    print("Paused after execution")
    print(f"Tool: {graph.state.paused_tool_name}")
    print(f"Result: {graph.state.paused_tool_result.result}")

    # Resume execution with the existing result
    await graph.resume(execute_tool=True)
```

### PostgreSQL Checkpoint with Tool Pauses

You can save and restore paused tool sessions using PostgreSQL:

```python
from primeGraph.checkpoint.postgresql import PostgreSQLStorage, PostgreSQLConfig

# Configure PostgreSQL storage
postgres_storage = PostgreSQLStorage.from_config(
    host="localhost",
    port=5432,
    user="primegraph",
    password="primegraph",
    database="primegraph",
)

# Create a tool graph with PostgreSQL storage
graph = ToolGraph(
    "payment_processing",
    state_class=ToolState,
    checkpoint_storage=postgres_storage  # Use PostgreSQL for checkpoints
)

# Add tools and node to graph
node = graph.add_tool_node(
    name="payment_agent",
    tools=[process_payment, update_account, get_user_details],
    llm_client=llm_client
)

# Connect nodes
graph.add_edge(START, node.name)
graph.add_edge(node.name, END)

# Run the graph - it will pause when a tool with pause flag is triggered
await graph.execute(initial_state)
chain_id = graph.chain_id  # Store this to load checkpoint later

# In a separate session or process, load the paused state
new_graph = ToolGraph(
    "payment_processing",
    state_class=ToolState,
    checkpoint_storage=postgres_storage
)

# Add the same tools and structure
# ...

# Load the checkpoint
new_graph.load_from_checkpoint(chain_id)

# Check if graph is paused
if new_graph.state.is_paused:
    # Resume execution from the paused state
    await new_graph.resume(execute_tool=True)  # approve
```

### Streaming with Redis

primeGraph supports real-time streaming of LLM responses using Redis, allowing you to build responsive applications that process AI outputs as they're generated.

```python
from primeGraph.graph.llm_clients import StreamingConfig, StreamingEventType

# Configure streaming with Redis
streaming_config = StreamingConfig(
    enabled=True,
    event_types={StreamingEventType.TEXT, StreamingEventType.CONTENT_BLOCK_STOP},
    redis_host="localhost",  # Docker-exposed Redis host
    redis_port=6379,         # Default Redis port
    redis_channel="my_stream_channel"  # Custom channel name
)

# Use the streaming config with your LLM-based workflows
# ...
```

You can consume these streaming events from any service or application that can connect to Redis:

```python
import redis
import json

# Connect to Redis
r = redis.Redis(host="localhost", port=6379)
pubsub = r.pubsub()

# Subscribe to your stream channel
pubsub.subscribe("my_stream_channel")

# Process incoming messages
for message in pubsub.listen():
    if message["type"] == "message":
        event = json.loads(message["data"])
        print(f"Received event: {event['type']}")

        if event["type"] == "text":
            print(event["text"], end="", flush=True)
```

This streaming capability is particularly useful for:

- Building responsive UIs that show real-time LLM thinking or generation
- Processing outputs incrementally without waiting for complete responses
- Creating multi-user applications where outputs need to be broadcast to multiple clients
- Implementing custom monitoring or logging of LLM interactions

For setting up Redis locally, refer to the [Docker setup instructions](docker/README.md).
