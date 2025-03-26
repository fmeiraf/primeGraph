#!/usr/bin/env python
"""
State-Aware Tools Example

This example demonstrates how to create tools that can directly access and modify
the graph state during execution, without using special case handlers in the engine.

Key concepts:
- Creating a custom state class that inherits from ToolState
- Defining tools that accept and modify state directly
- Using a single state object across multiple tool calls
- Tracking accumulated data through tool execution
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import Field
from rich import print as rprint

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv package not found. Environment variables must be set manually.")

from primeGraph.buffer.factory import History, LastValue
from primeGraph.constants import END, START
from primeGraph.graph.llm_clients import LLMClientFactory, Provider
from primeGraph.graph.llm_tools import (LLMMessage, ToolGraph, ToolLoopOptions,
                                        ToolState, tool)


# Define a custom state class that extends the base ToolState
class InventoryState(ToolState):
    """Custom state for inventory management with additional fields"""
    
    # Add custom fields to track inventory data
    inventory_items: History[Dict[str, Any]] = Field(default_factory=list)
    cart_items: History[Dict[str, Any]] = Field(default_factory=list)
    order_history: History[Dict[str, Any]] = Field(default_factory=list)
    current_user: LastValue[Optional[Dict[str, Any]]] = None
    total_transactions: LastValue[int] = 0
    last_action_timestamp: LastValue[Optional[str]] = None


# Define tools that can interact with the state
@tool("Get inventory item details")
async def get_inventory_item(item_id: str, state=None) -> Dict[str, Any]:
    """Get detailed information about an inventory item by ID"""
    # Simulate inventory database
    inventory_db = {
        "item-001": {
            "id": "item-001",
            "name": "Ergonomic Chair",
            "price": 299.99,
            "category": "Office Furniture",
            "in_stock": 15
        },
        "item-002": {
            "id": "item-002",
            "name": "Mechanical Keyboard",
            "price": 129.99,
            "category": "Computer Accessories",
            "in_stock": 42
        },
        "item-003": {
            "id": "item-003",
            "name": "Ultrawide Monitor",
            "price": 549.99,
            "category": "Displays",
            "in_stock": 7
        }
    }
    
    # Check if item exists
    if item_id not in inventory_db:
        return {"error": f"Item {item_id} not found"}
    
    item = inventory_db[item_id]
    
    # Update state if provided
    if state and hasattr(state, "inventory_items"):
        state.inventory_items.append(item)
        state.last_action_timestamp = datetime.now().isoformat()
    
    return item


@tool("Add item to cart")
async def add_to_cart(item_id: str, quantity: int, state=None) -> Dict[str, Any]:
    """Add an item to the shopping cart"""
    # Simulate inventory check
    inventory_db = {
        "item-001": {"id": "item-001", "name": "Ergonomic Chair", "price": 299.99, "in_stock": 15},
        "item-002": {"id": "item-002", "name": "Mechanical Keyboard", "price": 129.99, "in_stock": 42},
        "item-003": {"id": "item-003", "name": "Ultrawide Monitor", "price": 549.99, "in_stock": 7}
    }
    
    # Check if item exists
    if item_id not in inventory_db:
        return {"error": f"Item {item_id} not found"}
    
    # Check if enough stock
    if inventory_db[item_id]["in_stock"] < quantity:
        return {"error": f"Not enough stock. Requested: {quantity}, Available: {inventory_db[item_id]['in_stock']}"}
    
    item = inventory_db[item_id].copy()
    item["quantity"] = quantity
    item["total_price"] = quantity * item["price"]
    
    # Update state if provided
    if state and hasattr(state, "cart_items"):
        state.cart_items.append(item)
        state.last_action_timestamp = datetime.now().isoformat()
    
    return {
        "success": True,
        "message": f"Added {quantity} x {item['name']} to cart",
        "item": item
    }


@tool("Process order")
async def process_order(payment_method: str, state=None) -> Dict[str, Any]:
    """Process the current cart items as an order"""
    # Validate state
    if not state or not hasattr(state, "cart_items") or len(state.cart_items) == 0:
        return {"error": "No items in cart"}
    
    # Calculate order total
    cart_items = state.cart_items
    order_total = sum(item["total_price"] for item in cart_items)
    
    # Create order
    order = {
        "order_id": f"order-{len(state.order_history) + 1}",
        "items": cart_items,
        "total": order_total,
        "payment_method": payment_method,
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }
    
    # Update state
    state.order_history.append(order)
    state.cart_items = []  # Clear cart
    state.total_transactions += 1
    state.last_action_timestamp = datetime.now().isoformat()
    
    return {
        "success": True,
        "message": "Order processed successfully",
        "order": order
    }


@tool("Get order history")
async def get_order_history(state=None) -> Dict[str, Any]:
    """Get the user's order history"""
    if not state or not hasattr(state, "order_history"):
        return {"orders": []}
    
    return {
        "total_orders": len(state.order_history),
        "total_spent": sum(order["total"] for order in state.order_history),
        "orders": state.order_history
    }


@tool("Get cart summary")
async def get_cart_summary(state=None) -> Dict[str, Any]:
    """Get a summary of items in the cart"""
    if not state or not hasattr(state, "cart_items") or len(state.cart_items) == 0:
        return {"items": [], "total": 0.0}
    
    return {
        "items": state.cart_items,
        "item_count": len(state.cart_items),
        "total": sum(item["total_price"] for item in state.cart_items)
    }


async def run_inventory_example():
    """Run the inventory management example"""
    # Get API key from environment
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it or install python-dotenv and create a .env file with ANTHROPIC_API_KEY=your_key")
        sys.exit(1)
        
    # Create Anthropic client
    llm_client = LLMClientFactory.create_client(
        Provider.ANTHROPIC,
        api_key=anthropic_api_key
    )
    
    # Define tools to use
    inventory_tools = [
        get_inventory_item,
        add_to_cart,
        get_cart_summary,
        process_order,
        get_order_history
    ]
    
    # Create the ToolGraph with our custom state class
    graph = ToolGraph(
        name="inventory_manager", 
        state_class=InventoryState
    )
    
    # Add a tool node
    node = graph.add_tool_node(
        name="inventory_agent",
        tools=inventory_tools,
        llm_client=llm_client,
        options=ToolLoopOptions(max_iterations=10)
    )
    
    # Connect to START and END
    graph.add_edge(START, node.name)
    graph.add_edge(node.name, END)
    
    # Create initial state
    initial_state = InventoryState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are an inventory management assistant that helps users browse products, "
                    "add items to their cart, and process orders using the tools provided to you"
        ),
        LLMMessage(
            role="user",
            content="I'd like to buy an ergonomic chair and a mechanical keyboard."
        )
    ]
    
    # Execute the graph
    print("Executing inventory management workflow...")
    await graph.execute(initial_state=initial_state)
    
    # Access the final state
    final_state = graph.state
    
    # Print the results
    print("\n=== Final State Summary ===")
    print(f"Items viewed: {len(final_state.inventory_items)}")
    print(f"Items in cart: {len(final_state.cart_items)}")
    print(f"Orders placed: {len(final_state.order_history)}")
    print(f"Total transactions: {final_state.total_transactions}")
    
    if final_state.order_history:
        print("\n=== Last Order ===")
        last_order = final_state.order_history[-1]
        print(f"Order ID: {last_order['order_id']}")
        print(f"Total: ${last_order['total']:.2f}")
        print("Items:")
        for item in last_order['items']:
            print(f"  - {item['quantity']} x {item['name']} (${item['price']:.2f} each)")
    
    print("\n=== Final Messages ===")
    # Print the last few messages
    last_messages = final_state.messages[-3:] if len(final_state.messages) > 3 else final_state.messages
    for msg in last_messages:
        print(f"[{msg.role}]: {msg.content[:100]}..." if len(msg.content) > 100 else f"[{msg.role}]: {msg.content}")


    print('Final state:')
    rprint(final_state)
    
if __name__ == "__main__":
    asyncio.run(run_inventory_example()) 