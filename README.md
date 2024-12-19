<p align="center">
  <img src="docs/images/logo_art.png" alt="primeGraph Logo" width="200"/>
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

**primeGraph** is a Python library for building and executing workflow graphs, ranging from simple sequential processes to complex parallel execution patterns. While originally optimized for AI applications, its flexible architecture makes it suitable for any workflow orchestration needs.

Key principles:

- **Flexibility First**: Design your nodes and execution patterns with complete freedom.
- **Zero Lock-in**: Deploy and run workflows however you want, with no vendor dependencies.
- **Opinionated Yet Adaptable**: Structured foundations with room for customization.

_Note from the author: This project came to life through my experience of creating AI (wrapper) applications. I want to acknowledge [langgraph](https://www.langchain.com/langgraph) as the main inspiration for this project. As an individual developer, I wanted to gain experience creating my own workflow engine—one that's flexible enough to be deployed however you want, while opening doors for implementing more of my own ideas and learnings. This is a open source project though, so feel free to use it, modify it, and contribute to it._

#### Features

- **Flexible Graph Construction**: Build multiple workflows with sequential and parallel execution paths.
- **State Management**: Built-in state management with different buffer types to coordinate state management during workflow execution.
- **Type Safety**: Built-in type safety for your nodes' shared state using Pydantic.
- **Router Nodes**: Dynamic path selection based on node outputs.
- **Repeatable Nodes**: Execute nodes multiple times in parallel or sequence.
- **Subgraphs**: graphs can be composed of subgraphs to allow for more complex workflows.
- **Persistence**: Save and resume workflow execution using stored states (currently supports memory and Postgres).
- **Async Support**: Full async/await support for non-blocking execution.
- **Acyclical and Cyclical Graphs**: Build acyclical and cyclical graphs with ease.
- **Flow Control**: Support execution flow control for human-in-the-loop interactions.
- **Visualization**: Generate visual representations of your workflows with 0 effort.
- **Web Integration**: Built-in FastAPI integration with WebSocket support.
- **(Coming Soon) Streaming**: Stream outputs from your nodes as they are generated.

## Installation

## Usage

#### Basic Usage

```python
from primeGraph import Graph, GraphState
from primeGraph.buffer.factory import History, LastValue, Incremental

class DocumentProcessingState(GraphState):
    processed_files: History[str]  # Keeps track of all processed files
    current_status: LastValue[str]  # Current processing status
    total_processed: Incremental[int]  # Counter for processed documents

# Initialize state
state = DocumentProcessingState(
    processed_files=[],
    current_status="initializing",
    total_processed=0
)

# Create graph
graph = Graph(state=state)

@graph.node()
def load_documents(state):
    # Simulate loading documents
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "total_processed": 1
    }

@graph.node()
def validate_documents(state):
    # Validate loaded documents
    return {
        "current_status": "validating",
        "total_processed": 1
    }

@graph.node()
def process_documents(state):
    # Process documents
    return {
        "current_status": "completed",
        "total_processed": 1
    }

# Connect nodes
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "validate_documents")
graph.add_edge("validate_documents", "process_documents")
graph.add_edge("process_documents", END)

# Compile and execute
graph.compile()
graph.start()

```

#### Router Nodes

```python
@graph.node()
def route_documents(state):
    # Route based on document type
    if "invoice" in state.current_status:
        return "process_invoice"
    return "process_regular"

@graph.node()
def process_invoice(state):
    return {"current_status": "invoice_processed"}

@graph.node()
def process_regular(state):
    return {"current_status": "regular_processed"}

# Add router edges
graph.add_router_edge("validate_documents", "route_documents")
graph.add_edge("route_documents", "process_invoice", id="invoice_path")
graph.add_edge("route_documents", "process_regular", id="regular_path")
```

#### Repeatable Nodes

```python
@graph.node()
def process_batch(state):
    return {
        "processed_files": f"batch_{state.total_processed}",
        "total_processed": 1
    }

# Add repeating edge to process multiple batches
graph.add_repeating_edge(
    "load_documents",
    "process_batch",
    "validate_documents",
    repeat=3,  # Process 3 batches
    parallel=True  # Process in parallel
)

```

#### Subgraphs

```python
@graph.subgraph()
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

# Use subgraph in main flow
graph.add_edge("load_documents", "validation_subgraph")
graph.add_edge("validation_subgraph", "process_documents")

```

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

# Start execution
graph.start()

# Later, resume from checkpoint
graph.load_from_checkpoint(chain_id)
graph.resume()
```

#### Async Support

```python
@graph.node()
async def async_document_process(state):
    await asyncio.sleep(1)  # Simulate async processing
    return {
        "processed_files": "async_processed",
        "current_status": "async_complete"
    }

# Execute async graph
await graph.start_async()
```

#### Flow Control

#### Visualization

#### Web Integration

```python
from fastapi import FastAPI
from primeGraph.web import create_graph_service, wrap_graph_with_websocket

app = FastAPI()
graph_service = create_graph_service(
    graph=graph,
    checkpoint_storage=storage,
    path_prefix="/documents"
)

# Add routes to FastAPI app
app.include_router(graph_service.router)

# Wrap graph with WebSocket support
graph = wrap_graph_with_websocket(graph, graph_service)
```

## Roadmap

- [ ] Add streaming support
- [ ] Create documentation
- [ ] Add tools for agentic workflows
- [ ] Add inter node epheral state for short term interactions
- [ ] Add persistence support for other databases
