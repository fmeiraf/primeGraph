"""
Advanced example of LLM tool nodes in primeGraph.

This example demonstrates a more complex setup with:
1. Multiple tool nodes with different tools
2. Combined synchronous and asynchronous execution
3. Shared state between tool nodes
4. Tool execution checkpointing and resumability
"""

import asyncio
import os
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field
from rich import print as rprint

from primeGraph.buffer.factory import History
from primeGraph.graph.llm_clients import LLMClientFactory, Provider
from primeGraph.graph.llm_tools import (LLMMessage, ToolGraph, ToolLoopOptions,
                                        ToolState, tool)

load_dotenv()
# Custom state model with different sections for different phases
class ResearchPlannerState(ToolState):
    """State for research planning and execution with tools"""
    # Standard tool state fields are inherited
    
    # Planning phase
    research_topic: Optional[str] = None
    research_plan: Optional[str] = None
    search_queries: History[str] = Field(default_factory=list)
    
    # Research phase
    search_results: History[Dict] = Field(default_factory=list)
    retrieved_documents: History[Dict] = Field(default_factory=list)
    
    # Analysis phase
    key_insights: History[str] = Field(default_factory=list)
    summary: Optional[str] = None
    
    # Final output
    report: Optional[str] = None


# Define our tools

@tool("Create a research plan")
async def create_research_plan(topic: str, depth: int = 3) -> Dict:
    """
    Create a structured research plan for a topic.
    
    Args:
        topic: Research topic
        depth: Depth of research (1-5)
        
    Returns:
        Structured research plan
    """
    await asyncio.sleep(0.5)  # Simulate processing time
    
    # In a real implementation, this might involve more complex logic
    return {
        "topic": topic,
        "depth": depth,
        "suggested_queries": [
            f"{topic} overview",
            f"{topic} key concepts",
            f"{topic} recent developments",
            f"{topic} case studies"
        ],
        "estimated_time": f"{depth * 15} minutes"
    }


@tool("Search for information")
async def search_information(query: str, limit: int = 5) -> List[Dict]:
    """
    Search for information using a query.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of search results
    """
    await asyncio.sleep(0.7)  # Simulate API call
    
    # Mock search results
    base_results = [
        {
            "title": f"Introduction to {query}",
            "snippet": f"This article provides an overview of {query} and its importance.",
            "url": f"https://example.com/intro-{query.replace(' ', '-')}"
        },
        {
            "title": f"Advanced {query} techniques",
            "snippet": f"Explore advanced methods and techniques in {query} for professionals.",
            "url": f"https://example.com/advanced-{query.replace(' ', '-')}"
        },
        {
            "title": f"{query} case studies",
            "snippet": f"Real-world applications and case studies of {query} in various fields.",
            "url": f"https://example.com/cases-{query.replace(' ', '-')}"
        },
        {
            "title": f"History of {query}",
            "snippet": f"Learn about the historical development of {query} and its evolution.",
            "url": f"https://example.com/history-{query.replace(' ', '-')}"
        },
        {
            "title": f"Future trends in {query}",
            "snippet": f"Predictions and emerging trends in the field of {query} for the next decade.",
            "url": f"https://example.com/trends-{query.replace(' ', '-')}"
        },
        {
            "title": f"{query} for beginners",
            "snippet": f"A beginner's guide to understanding {query} concepts and terminology.",
            "url": f"https://example.com/beginners-{query.replace(' ', '-')}"
        },
        {
            "title": f"Comparing different approaches to {query}",
            "snippet": f"Analysis of various methodologies and approaches in {query}.",
            "url": f"https://example.com/comparison-{query.replace(' ', '-')}"
        }
    ]
    
    # Return limited results
    return base_results[:limit]


@tool("Retrieve document content")
async def retrieve_document(url: str) -> Dict:
    """
    Retrieve the content of a document by URL.
    
    Args:
        url: Document URL
        
    Returns:
        Document content and metadata
    """
    await asyncio.sleep(0.8)  # Simulate document retrieval
    
    
    
    # Generate mock content based on the URL
    content_parts = url.split('-')[1:]
    topic = ' '.join(content_parts).replace('/', ' ')
    
    if "intro" in url:
        content = f"""# Introduction to {topic}
        
This document provides a comprehensive introduction to {topic}, covering basic concepts, 
terminology, and fundamental principles. {topic.capitalize()} has become increasingly 
important in recent years due to technological advancements and changing paradigms in the field.
        
## Key Concepts
        
1. Definition and scope of {topic}
2. Historical context and development
3. Core principles and methodologies
4. Basic applications and use cases
        
## Importance
        
Understanding {topic} is crucial for professionals in related fields because it provides 
a foundation for more advanced concepts and applications. The fundamental principles of 
{topic} inform modern practices and innovation across multiple domains.
        """
    elif "advanced" in url:
        content = f"""# Advanced {topic} Techniques
        
This document explores cutting-edge techniques and methodologies in {topic} for experienced 
practitioners. These advanced approaches build upon fundamental concepts to provide more 
powerful, efficient, and effective solutions to complex problems.
        
## Advanced Methodologies
        
1. Optimization strategies for {topic} implementation
2. Integration of {topic} with complementary systems
3. Performance tuning and enhancement
4. Scaling {topic} for enterprise applications
        
## Case Studies
        
Several organizations have successfully implemented advanced {topic} techniques, resulting 
in significant improvements in efficiency, accuracy, and overall performance.
        """
    else:
        content = f"""# General Information on {topic}
        
This document contains general information about {topic}, including recent developments, 
applications, and research findings. The field of {topic} continues to evolve rapidly, 
with new discoveries and innovations emerging regularly.
        
## Current Landscape
        
The current state of {topic} reflects a dynamic and evolving field with contributions from 
diverse disciplines and perspectives. Recent research has expanded our understanding and 
opened new avenues for application and development.
        
## Future Directions
        
Emerging trends suggest that {topic} will continue to grow in importance and impact across 
various domains. Ongoing research and development promise to address current limitations and 
unlock new possibilities.
        """
    
    return {
        "url": url,
        "title": url.split('/')[-1].replace('-', ' ').title(),
        "content": content,
        "word_count": len(content.split()),
        "last_updated": "2025-03-10"
    }


@tool("Analyze document for key insights")
async def analyze_document(document: Dict, focus_areas: Optional[List[str]] = None) -> Dict:
    """
    Analyze a document for key insights.
    
    Args:
        document: Document content and metadata
        focus_areas: Optional specific areas to focus on
        
    Returns:
        Analysis results with key insights
    """
    await asyncio.sleep(1.0)  # Simulate analysis time
    
    content = document.get("content", "")
    title = document.get("title", "Unknown document")
    
    # Simple "analysis" based on content paragraphs
    paragraphs = content.split('\n\n')
    insights = []
    
    for para in paragraphs:
        if len(para.strip()) > 100:  # Only consider substantial paragraphs
            # Extract a "key insight" (just the first sentence in this mock version)
            first_sentence = para.strip().split('.')[0]
            if first_sentence and len(first_sentence) > 20:
                insights.append(first_sentence)
    
    return {
        "document_title": title,
        "key_insights": insights,
        "focus_areas_covered": focus_areas or ["general"],
        "analysis_confidence": 0.85
    }


@tool("Generate research summary")
async def generate_summary(insights: List[str], topic: str) -> Dict:
    """
    Generate a summary based on key insights.
    
    Args:
        insights: List of key insights
        topic: Research topic
        
    Returns:
        Summary information
    """
    await asyncio.sleep(1.2)  # Simulate generation time
    
    # In a real implementation, this would use more sophisticated summarization
    num_insights = len(insights)
    
    summary = f"""# Research Summary: {topic.capitalize()}

Based on the analysis of {num_insights} key insights, this research summary provides an overview 
of findings related to {topic}.

## Key Findings

"""
    
    for i, insight in enumerate(insights[:5]):
        summary += f"{i+1}. {insight}.\n\n"
        
    summary += f"""
## Conclusion

The research on {topic} reveals important patterns and information that contribute to a deeper
understanding of the subject. Further investigation may be warranted in specific areas to
develop a more comprehensive picture.
"""
    
    return {
        "topic": topic,
        "summary": summary,
        "insight_count": num_insights,
        "generated_at": time.time()
    }


async def main():
    # Create clients for the LLM providers
    openai_client = LLMClientFactory.create_client(
        Provider.OPENAI, 
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    anthropic_client = None
    try:
        anthropic_client = LLMClientFactory.create_client(
            Provider.ANTHROPIC, 
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    except ImportError:
        print("Anthropic client not available, using OpenAI for all nodes")
    
    # Use OpenAI client if Anthropic is not available
    second_client = anthropic_client or openai_client
    
    # Create a graph with our custom state
    graph = ToolGraph("research_assistant", state_class=ResearchPlannerState)
    
    # Create tool node options
    planner_options = ToolLoopOptions(
        max_iterations=3,
        max_tokens=1024
    )
    
    researcher_options = ToolLoopOptions(
        max_iterations=5,
        max_tokens=1024
    )
    
    analyzer_options = ToolLoopOptions(
        max_iterations=4,
        max_tokens=1024
    )
    
    # Create tool nodes
    planner_node = graph.add_tool_node(
        name="research_planner",
        tools=[create_research_plan],
        llm_client=openai_client,
        options=planner_options
    )
    
    researcher_node = graph.add_tool_node(
        name="researcher",
        tools=[search_information, retrieve_document],
        llm_client=openai_client,
        options=researcher_options
    )
    
    analyzer_node = graph.add_tool_node(
        name="analyzer",
        tools=[analyze_document, generate_summary],
        llm_client=second_client,
        options=analyzer_options
    )
    
    # Connect the nodes in sequence
    graph.add_edge(planner_node, researcher_node)
    graph.add_edge(researcher_node, analyzer_node)
    graph.add_edge(analyzer_node, graph.END)
    
    # Set up initial state with messages for the first node
    initial_state = ResearchPlannerState()
    initial_state.research_topic = "quantum computing"
    initial_state.messages = [
        LLMMessage(
            role="system",
            content="You are a research planning assistant. Help create a structured research plan for the given topic."
        ),
        LLMMessage(
            role="user",
            content="I need to research quantum computing. Please create a comprehensive research plan to help me understand the field, current developments, and future prospects."
        )
    ]
    
    # Execute the graph
    print("\nExecuting multi-step research workflow...")
    print("(This will use your OpenAI API key and may incur charges)")
    await graph.execute(initial_state=initial_state)
    
    # Access the final state
    final_state = graph.state
    
    # Print the final report
    print("\n=== Research Report ===")
    print(final_state.report or final_state.final_output or "No report generated.")
    
    print(f"\nTotal tool calls: {len(final_state.tool_calls)}")
    # Simplified analysis of tool usage
    tool_usage = {}
    for call in final_state.tool_calls:
        tool_usage[call.tool_name] = tool_usage.get(call.tool_name, 0) + 1
    
    print("\n=== Tool Usage ===")
    for tool, count in tool_usage.items():
        print(f"{tool}: {count} calls")
    
    rprint(final_state)


if __name__ == "__main__":
    asyncio.run(main())