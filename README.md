<p align="center">
  <img src="docs/images/logo_art.png" alt="primeGraph Logo" width="200"/>
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Package Version](https://img.shields.io/badge/package-0.1.0-blue.svg)](https://pypi.org/project/primegraph/)

---

## Overview

**primeGraph** is a Python library for building and executing workflows using graphs, ranging from simple sequential processes to complex parallel execution patterns. While originally optimized for AI applications, its flexible architecture makes it suitable for any workflow orchestration needs.

Key principles:

- **Flexibility First**: Design your nodes and execution patterns with complete freedom.
- **Zero Lock-in**: Deploy and run workflows however you want, with no vendor dependencies.
- **Opinionated Yet Adaptable**: Structured foundations with room for customization.

_Note from the author: This project came to life through my experience of creating AI applications. I want to acknowledge [langgraph](https://www.langchain.com/langgraph) as the main inspiration for this project. As an individual developer, I wanted to gain experience creating my own workflow engine to implement more of my own ideas and learnings. At the same time, I also wanted to create a framework that is flexible enough for others to deploy their apps however they want, as this is an open source project. So feel free to use it, modify it, and contribute to it._

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

```bash
pip install primeGraph
```

## Core Features

### The Basics

```python
from primeGraph import Graph
from primeGraph.models import GraphState
from primeGraph.buffer import History, LastValue, Incremental


# primeGraph uses the return values of the nodes to update the state (state is a pydantic model)
class DocumentProcessingState(GraphState):
    processed_files: History[str]  # History: stores all the values returned as a list
    current_status: LastValue[str]  # LastValue: keeps the last value returned
    number_of_executed_steps: Incremental[int]  # Incremental: increments the current value of the key by the returned value

# Initialize state
state = DocumentProcessingState(
    processed_files=[],
    current_status="initializing",
    number_of_executed_steps=0
)

# Create graph
graph = Graph(state=state)

#adding nodes to the graph
@graph.node()
def load_documents(state):
    # Simulate loading documents
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

@graph.node()
def validate_documents(state):
    # Validate loaded documents
    return {
        "current_status": "validating",
        "number_of_executed_steps": 1
    }

@graph.node()
def process_documents(state):
    # Process documents
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

# Connect nodes
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "validate_documents")
graph.add_edge("validate_documents", "process_documents")
graph.add_edge("process_documents", END)

# Compile and execute
graph.compile()
graph.start()

# state after execution
print(state)

# DocumentProcessingState(version='random_uuid',
#   processed_files=['document1.txt'],
#   current_status='completed',
#   number_of_executed_steps=3)

graph.visualize()
```

<p align="center">
  <img src="docs/images/readme_base_usage.png" alt="Basic Usage Graph Visualization" width="400"/>
</p>

### Router Nodes

```python
# previous Basic Usage ...example

@graph.node()
def load_documents(state):
    # Simulate loading documents
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

@graph.node()
def validate_documents(state):
    # Validate loaded documents
    return {
        "current_status": "validating",
        "number_of_executed_steps": 1
    }

@graph.node()
def process_documents(state):
    # Process documents
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

@graph.node()
def route_documents(state):
    # Route based on document type
    if "invoice" in state.current_status:
        return "process_invoice"
    return "cancel_invoice"

@graph.node()
def process_invoice(state):
    return {"current_status": "invoice_processed"}

@graph.node()
def cancel_invoice(state):
    return {"current_status": "invoice_cancelled"}

# Connect nodes
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "validate_documents")
graph.add_edge("validate_documents", "process_documents")


# Add router edges
graph.add_router_edge("process_documents", "route_documents")
graph.add_edge("process_invoice", END)
graph.add_edge("cancel_invoice", END)

# Compile and execute
graph.compile()
graph.start()

# state after execution
print(state)

# DocumentProcessingState(version='random_uuid',
#   processed_files=['document1.txt'],
#   current_status='invoice_cancelled',
#   number_of_executed_steps=4)

graph.visualize()
```

<p align="center">
  <img src="docs/images/readme_router_nodes.png" alt="Router Nodes visualization" width="400"/>
</p>

### Repeatable Nodes

```python
# previous Basic Usage ...example

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

# Add repeating edge to process multiple batches
graph.add_repeating_edge(
    "process_documents",
    "repeating_process_batch",
    "conclude_documents",
    repeat=3,
    parallel=True
)

graph.add_edge("conclude_documents", END)

# Compile and execute
graph.compile()
graph.start()

# state after execution
print(state)

# DocumentProcessingState(version='random_uuid',
# processed_files=['document1.txt', 'batch_3', 'batch_3', 'batch_5'],
# current_status='completed',
# number_of_executed_steps=7)

graph.visualize()
```

<p align="center">
  <img src="docs/images/readme_repeatable_nodes.png" alt="Repeatable Nodes visualization" width="400"/>
</p>

### Subgraphs

```python
# previous Basic Usage ...example

# Create graph
main_graph = Graph(state=state)

@main_graph.node()
def load_documents(state):
    # Simulate loading documents
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

# a subgbraph decorator is execting the function (which is now a new node) to return a subgraph
# you can either declare your subgraph in the function or reference from an existing subgraph
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
    # Process documents
    return {
        "current_status": "completed",
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
main_graph.add_edge("load_documents", "validation_subgraph") # subgreaph added as a normal node
main_graph.add_edge("load_documents", "pre_process_documents")
main_graph.add_edge("validation_subgraph", "conclude_documents")
main_graph.add_edge("pre_process_documents", "conclude_documents")
main_graph.add_edge("conclude_documents", END)

# Compile and execute
main_graph.compile()
main_graph.start()

# state after execution
print(state)

# DocumentProcessingState(version='random_uuid',
# processed_files=['document1.txt'],
# current_status='completed',
# number_of_executed_steps=3)

graph.visualize()
```

<p align="center">
  <img src="docs/images/readme_subgraphs.png" alt="Subgraphs visualization" width="400"/>
</p>

### Flow Control

```python
# previous Basic Usage ...example

# Create graph
graph = Graph(state=state)

@graph.node()
def load_documents(state):
    # Simulate loading documents
    return {
        "processed_files": "document1.txt",
        "current_status": "loading",
        "number_of_executed_steps": 1
    }

# using interrupt="before" will interrupt the execution before this node is executed
# using interrupt="after" will interrupt the execution after this node is executed
@graph.node(interrupt="before")
def review_documents(state):
    # Validate loaded documents
    return {
        "current_status": "validating",
        "number_of_executed_steps": 1
    }

@graph.node()
def process_documents(state):
    # Process documents
    return {
        "current_status": "completed",
        "number_of_executed_steps": 1
    }

# Connect nodes
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "review_documents")
graph.add_edge("review_documents", "process_documents")
graph.add_edge("process_documents", END)

# Compile and execute
graph.compile()
graph.start()


# state until interrupted
print(state)

# DocumentProcessingState(version='random_uuid',
#   processed_files=['document1.txt'],
#   current_status='loading',
#   number_of_executed_steps=1)


graph.resume()

# state after finishing
print(state)

# DocumentProcessingState(version='random_uuid',
#   processed_files=['document1.txt'],
#   current_status='completed',
#   number_of_executed_steps=3)

graph.visualize()
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

# Start execution
chain_id = graph.start()

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

# Resume async graph
await graph.resume_async()
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
from primeGraph import Graph, END, START
from primeGraph.models import GraphState
from primeGraph.web import create_graph_service, wrap_graph_with_websocket

logging.basicConfig(level=logging.DEBUG)

# Create FastAPI app
app = FastAPI()


# Explicitly set logging levels for key loggers
logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("fastapi").setLevel(logging.DEBUG)
logging.getLogger("websockets").setLevel(logging.DEBUG)
logging.getLogger("primeGraph").setLevel(logging.DEBUG)

# Your existing imports...

app = FastAPI(debug=True)  # Enable debug mode

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Your existing routes
@app.get("/hello")
async def hello():
    return {"message": "Hello World"}


# Create multiple graphs if needed
graphs: List[Graph] = []


# Define state model
class SimpleGraphState(GraphState):
    messages: History[str]


# Create state instance
state = SimpleGraphState(messages=[])

# Update graph with state
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


# Add edges
graph1.add_edge(START, "add_hello")
graph1.add_edge("add_hello", "add_world")
graph1.add_edge("add_world", "add_exclamation")
graph1.add_edge("add_exclamation", END)

# Add nodes and edges...
graph1.compile()


# Create graph service
service = create_graph_service(graph1, storage, path_prefix="/graphs/workflow1")


# Include the router in your app
app.include_router(service.router, tags=["workflow1"])



# access your graph at http://localhost:8000/graphs/workflow1/
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```

## Basic Usage examples

_Find examples in the [examples](examples) folder._

### Chatbot (yep, one more chatbot example)

```python
from primeGraph import Graph
from primeGraph.models import GraphState
from primeGraph.buffer import History, LastValue, Incremental
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor

class ChatbotState(GraphState):
    chat_history: History[dict[str, str]]
    user_wants_to_exit: LastValue[bool] = Field(default=False)

class ChatbotResponse(BaseModel):
    chat_message: str
    user_requested_to_quit: bool = Field(description="returns true if user is requesting to quit the chat")


chatbot_state = ChatbotState(chat_history=[], user_wants_to_exit=False)
chatbot_graph = Graph(state=chatbot_state, verbose=False)

@chatbot_graph.node(interrupt="before")
def chat_with_user(state):

    # user input will be inserted directly into the chat_history on the state
    # Extract structured data from natural language
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=ChatbotResponse,
            messages=state.chat_history,
        )
        print(res.chat_message)
        return {"chat_history": {"role": "assistant", "content": res.chat_message},
                "user_wants_to_exit": res.user_requested_to_quit}

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
```

<p align="center">
  <img src="docs/images/readme_chatbot.png" alt="Chatbot visualization" width="400"/>
</p>

```python
# Running the chatbot on a loop
chatbot_graph.start()

def add_user_message(message: str):
    chatbot_state.chat_history.append({"role": "user", "content": message})

while not chatbot_state.user_wants_to_exit:

    user_input = input("Your message: ")
    print(f"You: {user_input}")
    add_user_message(user_input)

    chatbot_graph.resume()

print("Bye")


```

### Async workflow

```python
from primeGraph import Graph, START, END
from primeGraph.models import GraphState
from primeGraph.buffer import History, LastValue
from pydantic import BaseModel
from openai import AsyncOpenAI
import instructor
from IPython.display import Image
from typing import Tuple

from dotenv import load_dotenv

# assumes you have a local .env file with OPENAI_API_KEY set
load_dotenv()

# loading openai client
client = instructor.from_openai(AsyncOpenAI())

class Character(GraphState):
    character_name: LastValue[str]
    character_items: History[Tuple[str,str]]
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
    res = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Pick me a character from Lord of the Rings"}],
        response_model=CharacterName,
    )
    return {"character_name": res.character_name}


@character_graph.node()
async def pick_character_profession(state):
    res = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Pick me a profession for the character"}],
        response_model=CharacterItem,
    )
    return {"character_items": (res.item_name, res.item_description)}

@character_graph.node()
async def pick_character_apparel(state):
    res = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Pick me a clothing for the character"}],
        response_model=CharacterItem,
    )
    return {"character_items": (res.item_name, res.item_description)}

@character_graph.node()
async def pick_character_partner(state):
    res = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Pick me a partner for the character"}],
        response_model=CharacterItem,
    )
    return {"character_items": (res.item_name, res.item_description)}

@character_graph.node()
async def create_charater_summary(state):
    ch_items = "\n".join([f"{item[0]}: {item[1]}" for item in state.character_items])
    res = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Name: {state.character_name} \
        \nItems: {ch_items}"}],
        response_model=CharacterSummary,
    )
    return {"character_summary": res.character_summary}

character_graph.add_edge(START, "pick_character_name")

# setting tasks to run in parallel
character_graph.add_edge("pick_character_name", "pick_character_profession")
character_graph.add_edge("pick_character_name", "pick_character_apparel")
character_graph.add_edge("pick_character_name", "pick_character_partner")

character_graph.add_edge("pick_character_profession", "create_charater_summary")
character_graph.add_edge("pick_character_apparel", "create_charater_summary")
character_graph.add_edge("pick_character_partner", "create_charater_summary")
character_graph.add_edge("create_charater_summary", END)

character_graph.compile()
Image(character_graph.visualize(transparent=False).pipe(format='png'))

```

<p align="center">
  <img src="docs/images/readme_async_workflow.png" alt="Async Workflow visualization" width="400"/>
</p>

```python
from rich import print as rprint

await character_graph.start_async()
rprint(character_graph.state)

# Character(
#     version='a35efff8c805417e13d4b950e6d7281c',
#     character_name='Frodo Baggins',
#     character_items=[
#         (
#             'Mysterious Stranger',
#             "A hooded figure who appears at unexpected moments, offering cryptic advice and insight into the
# character's quest."
#         ),
#         (
#             'Mystic Robe',
#             "A flowing robe made from shimmering fabric that glimmers with magical energy. It is adorned with
# ancient runes and has a hood that conceals the wearer's face. Perfect for wizards and sorcerers."
#         ),
#         (
#             'Adventurer',
#             'A brave explorer who embarks on quests, seeks treasure, and faces challenges in the great unknown.'
#         )
#     ],
#     character_summary='Frodo Baggins is a brave adventurer on a quest, known for exploring the unknown and seeking
# treasure. He is accompanied by a Mysterious Stranger, a hooded figure who offers cryptic advice and insight during
# his journey. Frodo wears a Mystic Robe, a magical garment adorned with ancient runes, which enhances his mystical
# abilities and conceals his identity.'
# )
```

## Roadmap

- [ ] Add streaming support
- [ ] Create documentation
- [ ] Add tools for agentic workflows
- [ ] Add inter node epheral state for short term interactions
- [ ] Add persistence support for other databases

```

```
