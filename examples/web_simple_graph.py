import os
import sys

sys.path.append(os.path.abspath(".."))

import logging
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from primeGraph.buffer.factory import History
from primeGraph.checkpoint.local_storage import LocalStorage
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState
from primeGraph.web import create_graph_service
from primeGraph.web.graph_wrapper import wrap_graph_with_websocket

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

# Setup checkpoint storage


# Create graph service
service = create_graph_service(graph1, storage, path_prefix="/graphs/workflow1")

# Wrap graph with WebSocket support
# graph1 = wrap_graph_with_websocket(graph1, service) # TODO: this websocket wrapper is not working

# Include the router in your app
app.include_router(service.router, tags=["workflow1"])


# Add endpoints to inspect state and storage
@app.get("/graphs/workflow1/state", tags=["workflow1"])
async def get_state():
    print(state)
    return {"messages": state.messages}


@app.get("/graphs/workflow1/storage", tags=["workflow1"])
async def get_storage():
    # Get all chain IDs and their checkpoints
    print(storage._storage)
    chain_data = {}
    for key, value in storage._storage.items():
        chain_data[key] = value
    return chain_data


# Add another graph if needed
# Create state instance
state2 = SimpleGraphState(messages=[])

# Update graph with state
storage2 = LocalStorage()
graph2 = Graph(state=state2, checkpoint_storage=storage2)


@graph2.node()
async def step1(state: GraphState):
    logging.debug("step1")
    return {"messages": "Hello"}


@graph2.node(interrupt="before")
async def step2(state: GraphState):
    await step2.emit_event("step2_event", {"messages": "World"})
    logging.debug("step2")
    logging.debug("we stop")
    return {"messages": "World"}


@graph2.node()
async def step3(state: GraphState):
    await step3.emit_event("step3_event", {"messages": "!"})
    logging.debug("step3")
    return {"messages": "!"}


# Add edges
graph2.add_edge(START, "step1")
graph2.add_edge("step1", "step2")
graph2.add_edge("step2", "step3")
graph2.add_edge("step3", END)

# Add nodes and edges...
graph2.compile()


# Add endpoints to inspect state and storage
@app.get("/graphs/workflow2/state", tags=["workflow2"])
async def get_state2():
    return {"messages": state2.messages}


@app.get("/graphs/workflow2/storage", tags=["workflow2"])
async def get_storage2():
    # Get all chain IDs and their checkpoints

    chain_data = {}
    for key, value in storage2._storage.items():
        chain_data[key] = value
    return chain_data


# Create graph service

service2 = create_graph_service(graph2, storage2, path_prefix="/graphs/workflow2")
graph2 = wrap_graph_with_websocket(graph2, service2)
app.include_router(service2.router, tags=["workflow2"])


@app.get("/debug/routes")
async def debug_routes():
    routes = []
    for route in app.routes:
        routes.append(
            {
                "path": route.path,
                "name": route.name,
                "methods": getattr(route, "methods", None),
            }
        )
    return routes


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
