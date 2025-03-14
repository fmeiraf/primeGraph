"""
LLM Client interfaces for tool execution.

This module provides client interfaces for interacting with different LLM providers,
specifically focusing on tool/function calling capabilities.
"""

import asyncio
import json
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import openai


class Provider(str, Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMClientBase:
    """
    Base class for LLM clients that support tool/function calling.

    This abstract class defines the interface that all provider-specific
    clients must implement.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.
        Args:
            api_key: API key for the provider. If None, will try to get from environment.
        """
        self.api_key = api_key

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, bool, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Generate a response using the LLM, possibly using tools.

        Args:
            messages: List of messages for the conversation
            tools: Optional list of tool definitions
            tool_choice: Optional specification for tool choice behavior
            **kwargs: Additional parameters for the API

        Returns:
            A tuple of (response_text, raw_response)
        """
        raise NotImplementedError("Subclasses must implement this method")

    def is_tool_use_response(self, response: Any) -> bool:
        """
        Check if the response contains a tool use request.

        Args:
            response: Raw response from the LLM API

        Returns:
            True if the response contains tool calls, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the response.

        Args:
            response: Raw response from the LLM API

        Returns:
            List of dictionaries with tool call information
        """
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIClient(LLMClientBase):
    """Client for OpenAI models with function calling support."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client."""
        super().__init__(api_key)
        # Lazy import to avoid dependency issues if not using OpenAI
        self.client: Optional[openai.OpenAI] = None
        try:
            self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            self.client = None

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, bool, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Generate a response using OpenAI's API."""
        if self.client is None:
            raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'")

        # Process tool_choice into OpenAI format
        formatted_tool_choice = None
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                # Force use of a specific tool
                formatted_tool_choice = {"type": "function", "function": {"name": tool_choice}}
            elif isinstance(tool_choice, bool):
                # True means "any tool", False means "auto"
                formatted_tool_choice = {"type": "any" if tool_choice else "auto"}
            elif isinstance(tool_choice, dict):
                # Pass through existing tool_choice dictionary
                formatted_tool_choice = tool_choice

        api_kwargs = {**kwargs}
        if tools:
            api_kwargs["tools"] = tools
        if formatted_tool_choice:
            api_kwargs["tool_choice"] = formatted_tool_choice

        # Ensure a model is specified - use GPT-4 by default for tool calling
        if "model" not in api_kwargs:
            api_kwargs["model"] = "gpt-4-turbo"

        # Call the API in a non-blocking way
        response = await asyncio.to_thread(self.client.chat.completions.create, messages=messages, **api_kwargs)  # type: ignore

        # Extract the content from the response
        content = response.choices[0].message.content or ""

        return content, response

    def is_tool_use_response(self, response: Any) -> bool:
        """Check if response requires tool use."""
        message = response.choices[0].message
        return hasattr(message, "tool_calls") and message.tool_calls

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenAI response."""
        tool_calls = []
        message = response.choices[0].message

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                # Parse arguments - OpenAI provides them as a JSON string
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {"input": tool_call.function.arguments}

                tool_calls.append({"id": tool_call.id, "name": tool_call.function.name, "arguments": args})

        return tool_calls


class AnthropicClient(LLMClientBase):
    """Client for Anthropic Claude models with tool use support."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Anthropic client."""
        super().__init__(api_key)
        # Lazy import to avoid dependency issues if not using Anthropic
        try:
            import anthropic  # type: ignore

            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            self.client = None  # type: ignore

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, bool, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Generate a response using Anthropic's API."""
        if self.client is None:
            raise ImportError("Anthropic package is not installed. Install it with 'pip install anthropic'")

        # Anthropic requires system messages to be passed separately
        anthropic_messages = []
        system_content = None

        # Extract system message and clean up all messages for Anthropic format
        for msg in messages:
            # Get the essential fields
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle system messages separately
            if role == "system":
                system_content = content
                continue

            # Convert 'tool' role to 'user' for Anthropic since it only supports user/assistant
            if role == "tool":
                role = "user"

            # Create a clean message with only the fields Anthropic accepts
            clean_msg = {"role": role, "content": content}

            # Only include messages with supported roles
            if role in ["user", "assistant"]:
                anthropic_messages.append(clean_msg)

        # Process tool_choice for Anthropic format
        anthropic_tool_choice = None
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                # Use a specific tool
                anthropic_tool_choice = tool_choice
            elif isinstance(tool_choice, bool):
                # True enables tool use, False/None lets model decide
                anthropic_tool_choice = tool_choice if tool_choice else None  # type: ignore
            elif isinstance(tool_choice, dict):
                if tool_choice.get("type") == "function":
                    anthropic_tool_choice = tool_choice["function"]["name"]
                elif tool_choice.get("type") == "any":
                    anthropic_tool_choice = True  # type: ignore
                else:
                    anthropic_tool_choice = None

        api_kwargs = {**kwargs}
        if tools:
            api_kwargs["tools"] = tools
        if anthropic_tool_choice is not None:
            api_kwargs["tool_choice"] = anthropic_tool_choice
        if system_content:
            api_kwargs["system"] = system_content

        # Ensure a model is specified - use Claude 3 by default for tool calling
        if "model" not in api_kwargs:
            api_kwargs["model"] = "claude-3-7-sonnet-latest"

        # Ensure max_tokens is set
        if "max_tokens" not in api_kwargs:
            api_kwargs["max_tokens"] = 4096

        # Call the API in a non-blocking way
        response = await asyncio.to_thread(self.client.messages.create, messages=anthropic_messages, **api_kwargs)  # type: ignore

        # Extract and join text from response content blocks
        content = ""
        if hasattr(response, "content") and response.content:
            if isinstance(response.content, list):
                content = "".join(
                    block.text if hasattr(block, "text") else str(block)
                    for block in response.content
                    if getattr(block, "type", None) != "tool_use"
                )
            else:
                content = response.content

        return content, response

    def is_tool_use_response(self, response: Any) -> bool:
        """Check if response requires tool use."""
        if hasattr(response, "content") and isinstance(response.content, list):
            return any(getattr(block, "type", None) == "tool_use" for block in response.content)
        return False

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Anthropic response."""
        tool_calls = []

        if hasattr(response, "content") and isinstance(response.content, list):
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_call = {
                        "id": getattr(block, "id", f"tool_{int(time.time())}"),
                        "name": getattr(block, "name", ""),
                        "arguments": getattr(block, "input", {}),
                    }
                    tool_calls.append(tool_call)

        return tool_calls


class LLMClientFactory:
    """Factory to create appropriate clients for each LLM provider."""

    @staticmethod
    def create_client(provider: Provider, api_key: Optional[str] = None) -> LLMClientBase:
        """
        Create a client for the specified provider.

        Args:
            provider: Provider enum value
            api_key: Optional API key

        Returns:
            Client instance for the provider

        Raises:
            ValueError: If provider is not supported
        """
        if provider == Provider.OPENAI:
            return OpenAIClient(api_key)
        elif provider == Provider.ANTHROPIC:
            return AnthropicClient(api_key)
        elif provider == Provider.GOOGLE:
            raise NotImplementedError("Google AI client is not yet implemented")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
