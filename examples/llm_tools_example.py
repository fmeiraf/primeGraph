"""
Example of using the LLM tool nodes in primeGraph.

This example demonstrates how to create and execute a graph with tool nodes
that interact with LLMs and execute tool functions.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field
from rich import print as rprint

from primeGraph import END, START
from primeGraph.buffer.factory import LastValue
from primeGraph.graph.llm_clients import LLMClientFactory, Provider
from primeGraph.graph.llm_tools import (LLMMessage, ToolGraph, ToolLoopOptions,
                                        ToolState, tool)

# Load environment variables for API keys
load_dotenv()


# Example state model with custom fields
class CustomerServiceState(ToolState):
    """State for customer service agent tools"""
    customer_data: LastValue[Optional[Dict]] = Field(default=None)
    order_data: LastValue[Optional[Dict]] = Field(default=None)
    cancelled_orders: LastValue[List[str]] = Field(default_factory=list)


# Define tools using the @tool decorator
@tool("Get customer information")
async def get_customer_info(customer_id: str, state=None) -> Dict:
    """
    Get customer details by ID
    
    Args:
        customer_id: Customer identifier
        state: Optional state object to update
        
    Returns:
        Customer information
    """
    # Test data
    customers = {
        "C1": {
            "id": "C1", 
            "name": "John Doe", 
            "email": "john@example.com",
            "orders": ["O1", "O2"]
        },
        "C2": {
            "id": "C2", 
            "name": "Jane Smith", 
            "email": "jane@example.com",
            "orders": ["O3"]
        }
    }
    
    if customer_id not in customers:
        raise ValueError(f"Customer {customer_id} not found")
    
    customer_data = customers[customer_id]
    
    # Update the state directly if provided
    if state is not None and hasattr(state, "customer_data"):
        state.customer_data = customer_data
    
    return customer_data


@tool("Get order details")
async def get_order_details(order_id: str, state=None) -> Dict:
    """
    Get order details by ID
    
    Args:
        order_id: Order identifier
        state: Optional state object to update
        
    Returns:
        Order information
    """
    # Test data
    orders = {
        "O1": {
            "id": "O1",
            "customer_id": "C1",
            "product": "Widget A",
            "quantity": 2,
            "price": 19.99,
            "status": "shipped"
        },
        "O2": {
            "id": "O2",
            "customer_id": "C1",
            "product": "Gadget B",
            "quantity": 1,
            "price": 49.99,
            "status": "processing"
        },
        "O3": {
            "id": "O3",
            "customer_id": "C2",
            "product": "Gizmo C",
            "quantity": 3,
            "price": 29.99,
            "status": "delivered"
        }
    }
    
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    
    order_data = orders[order_id]
    
    # Update the state directly if provided
    if state is not None and hasattr(state, "order_data"):
        state.order_data = order_data
    
    return order_data


@tool("Cancel an order")
async def cancel_order(order_id: str, state=None) -> Dict:
    """
    Cancel an order by ID
    
    Args:
        order_id: Order identifier
        state: Optional state object to update
        
    Returns:
        Cancellation result
    """
    # Test data
    orders = {
        "O1": {
            "id": "O1",
            "customer_id": "C1",
            "product": "Widget A",
            "status": "shipped"
        },
        "O2": {
            "id": "O2",
            "customer_id": "C1",
            "product": "Gadget B",
            "status": "processing"
        }
    }
    
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    
    # Update status to cancelled
    result = orders[order_id].copy()
    result["status"] = "cancelled"
    
    # Update cancelled_orders in state if provided
    if state is not None and hasattr(state, "cancelled_orders"):
        if order_id not in state.cancelled_orders:
            state.cancelled_orders.append(order_id)
    
    return {
        "order_id": order_id,
        "status": "cancelled",
        "message": f"Order {order_id} has been cancelled successfully",
        "order_details": result
    }


@tool("Process payment", pause_before_execution=True)
async def process_payment(order_id: str, amount: float, state=None) -> Dict:
    """
    Process a payment for an order, pausing for verification
    
    Args:
        order_id: Order identifier
        amount: Payment amount
        state: Optional state object to update
        
    Returns:
        Payment confirmation
    """
    # This would normally interact with a payment gateway
    # but for demonstration it just returns a confirmation
    await asyncio.sleep(0.5)  # Simulate payment processing time
    
    result = {
        "order_id": order_id,
        "amount": amount,
        "status": "processed",
        "transaction_id": f"TX-{order_id}-{int(time.time())}"
    }
    
    # Store the result in a processing_results list in the state
    if state is not None:
        # Create a processing_results field if it doesn't exist
        if not hasattr(state, "processing_results"):
            state.processing_results = []
        
        # Append the result
        state.processing_results.append(result)
    
    return result


async def main():
    # Get API key from environment variables
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return
        
    # Create a client for the LLM provider
    anthropic_client = LLMClientFactory.create_client(
        Provider.ANTHROPIC, 
        api_key=api_key
    )
    
    # Create a graph with our custom state
    graph = ToolGraph("customer_service", state_class=CustomerServiceState)
    
    # Create tool options
    options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024,
        stop_on_first_error=True
    )
    
    # Add a tool node with our tools
    tool_node = graph.add_tool_node(
        name="customer_service_agent",
        tools=[get_customer_info, get_order_details, cancel_order, process_payment],
        llm_client=anthropic_client,
        options=options
    )
    
    # Connect the node to the start and end
    graph.add_edge(START, tool_node.name)
    graph.add_edge(tool_node.name, END)
    
    # Set up initial messages in the state
    initial_state = CustomerServiceState()
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a helpful customer service assistant. Use the provided tools to help customers with their orders."
        ),
        LLMMessage(
            role="user",
            content="Please cancel all orders for customer C1."
        )
    ]
    
    # Execute the graph
    print("\nRunning the customer service assistant...")
    print("(This will use your Anthropic API key and may incur charges)")
    await graph.execute(initial_state=initial_state)
    
    # Access the final state
    final_state = graph.state

    # Print results
    print("\n=== Final Output ===")
    print(final_state.final_output)
    
    print("\n=== Tool Calls ===")
    for i, tool_call in enumerate(final_state.tool_calls):
        print(f"\nTool Call {i+1}:")
        print(f"  Tool: {tool_call.tool_name}")
        print(f"  Arguments: {json.dumps(tool_call.arguments, indent=2)}")
        print(f"  Result: {json.dumps(tool_call.result, indent=2) if tool_call.result else 'N/A'}")
    
    # Print the custom state fields
    print("\n=== Custom State Data ===")
    print(f"Customer Data: {json.dumps(final_state.customer_data, indent=2) if final_state.customer_data else 'None'}")
    print(f"Order Data: {json.dumps(final_state.order_data, indent=2) if final_state.order_data else 'None'}")
    print(f"Cancelled Orders: {final_state.cancelled_orders}")
    if hasattr(final_state, "processing_results"):
        print(f"Processing Results: {json.dumps(final_state.processing_results, indent=2)}")

    rprint(final_state)


if __name__ == "__main__":
    asyncio.run(main())