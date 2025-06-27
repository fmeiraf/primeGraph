"""
LLM Client interfaces for tool execution.

This module provides client interfaces for interacting with different LLM providers,
specifically focusing on tool/function calling capabilities.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import openai
from pydantic import BaseModel, Field

# Type definitions for Anthropic
# This helps with mypy's type checking without requiring the dependency
MessageParam = Dict[str, Any]  # Type alias for Anthropic message parameters
MessageParamT = TypeVar("MessageParamT", bound=MessageParam)

logger = logging.getLogger(__name__)


class LLMMessage(BaseModel):
    """Message in an LLM conversation"""

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    id: Optional[str] = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    # Alternative timestamp-based ID: Field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    should_show_to_user: bool = True  # Flag to indicate if this message should be shown to the user
    type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {
        "extra": "allow"  # Allow additional fields not specified in the model
    }


class Provider(str, Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class StreamingEventType(str, Enum):
    """Types of streaming events"""

    TEXT = "text"
    INPUT_JSON = "input_json"
    MESSAGE_STOP = "message_stop"
    CONTENT_BLOCK_STOP = "content_block_stop"
    TOOL_USE = "tool_use"
    MESSAGE_START = "message_start"
    ALL = "all"


class StreamingConfig(BaseModel):
    """Configuration for streaming responses"""

    enabled: bool = False
    event_types: Set[StreamingEventType] = Field(default_factory=lambda: {StreamingEventType.TEXT})
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_channel: Optional[str] = None
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    event_type_mapping: Dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Handle ALL event type
        if StreamingEventType.ALL in self.event_types:
            self.event_types = set(StreamingEventType)
            self.event_types.remove(StreamingEventType.ALL)

        # Validate configuration
        if self.enabled and not ((self.redis_channel and self.redis_host) or self.callback):
            raise ValueError(
                "When streaming is enabled, you must provide either redis_host and redis_channel "
                "or a callback function"
            )


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
        streaming_config: Optional[StreamingConfig] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Generate a response using the LLM, possibly using tools.

        Args:
            messages: List of messages for the conversation
            tools: Optional list of tool definitions
            tool_choice: Optional specification for tool choice behavior
            streaming_config: Optional configuration for streaming responses
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
        self.provider = "openai"
        try:
            self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            self.client = None

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        streaming_config: Optional[StreamingConfig] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Generate a response using OpenAI's API."""
        if self.client is None:
            raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'")

        api_kwargs = {**kwargs}
        if tools:
            api_kwargs["tools"] = tools
        api_kwargs["tool_choice"] = "auto"

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
        tool_objects = []
        message = response.choices[0].message

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_objects.append(tool_call.model_dump())
                # Parse arguments - OpenAI provides them as a JSON string
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {"input": tool_call.function.arguments}

                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": args,
                        "type": tool_call.type,
                    }
                )

        tool_message = LLMMessage(
            role="assistant",
            tool_calls=tool_objects,
        )

        return tool_calls, tool_message


class AnthropicClient(LLMClientBase):
    """Client for Anthropic Claude models with tool use support."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Anthropic client."""
        super().__init__(api_key)
        # Lazy import to avoid dependency issues if not using Anthropic
        self.provider = "anthropic"
        try:
            import anthropic  # type: ignore

            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            self.client = None  # type: ignore
            self.async_client = None  # type: ignore

    async def _publish_event(self, event: Dict[str, Any], streaming_config: StreamingConfig) -> None:
        """
        Publish a streaming event using the configured method (Redis or callback).

        Args:
            event: The event data to publish
            streaming_config: The streaming configuration
        """
        # Add timestamp
        event["timestamp"] = time.time()

        # Apply custom event type mapping if configured
        if streaming_config.event_type_mapping and event.get("type") in streaming_config.event_type_mapping:
            event["type"] = streaming_config.event_type_mapping[event["type"]]

        # Use callback if provided
        if streaming_config.callback:
            try:
                streaming_config.callback(event)
            except Exception as e:
                print(f"Error in streaming callback: {e}")

        # Publish to Redis if configured
        if streaming_config.redis_host and streaming_config.redis_channel:
            try:
                import redis

                r = redis.Redis(host=streaming_config.redis_host, port=streaming_config.redis_port)
                r.publish(streaming_config.redis_channel, json.dumps(event))
            except ImportError:
                print("Redis package not installed. Install with 'pip install redis'")
            except Exception as e:
                print(f"Error publishing to Redis: {e}")

    async def _handle_stream(self, stream: Any, streaming_config: StreamingConfig) -> Tuple[str, Any]:
        """
        Handle streaming response from Anthropic.

        Args:
            stream: The message stream from Anthropic
            streaming_config: Streaming configuration

        Returns:
            Tuple of (content, full_response)
        """
        content_parts = []
        full_response_metadata = {}  # To store metadata from message_start

        async for event in stream:
            # Process based on event type
            if event.type == "message_start" and StreamingEventType.MESSAGE_START in streaming_config.event_types:
                # Extract serializable usage data
                usage_data = {}
                if hasattr(event.message, "usage") and event.message.usage:
                    # Attempt to convert usage object to dict
                    if hasattr(event.message.usage, "_asdict"):
                        usage_data = event.message.usage._asdict()
                    elif isinstance(event.message.usage, dict):
                        usage_data = event.message.usage
                    else:
                        try:
                            # Fallback conversion if needed
                            usage_data = dict(event.message.usage)
                        except TypeError:
                            usage_data = {"raw": str(event.message.usage)}  # Store as string if fails

                # Store metadata for potential later use, though response might overwrite it
                full_response_metadata = {
                    "id": getattr(event.message, "id", None),
                    "model": getattr(event.message, "model", None),
                    "role": getattr(event.message, "role", None),
                    "type": getattr(event.message, "type", None),
                    "usage": usage_data,
                }
                await self._publish_event(
                    {
                        "type": "message_start",
                        "message": full_response_metadata,  # Publish the extracted metadata
                    },
                    streaming_config,
                )

            elif event.type == "text" and StreamingEventType.TEXT in streaming_config.event_types:
                content_parts.append(event.text)
                await self._publish_event(
                    {
                        "type": "text",
                        "text": event.text,
                        "snapshot": event.snapshot,
                    },
                    streaming_config,
                )

            elif (
                event.type == "content_block_stop"
                and StreamingEventType.CONTENT_BLOCK_STOP in streaming_config.event_types
            ):
                await self._publish_event(
                    {
                        "type": "content_block_stop",
                        "content_block": {
                            "type": getattr(event.content_block, "type", None),
                            "text": getattr(event.content_block, "text", None)
                            if hasattr(event.content_block, "text")
                            else None,
                        },
                    },
                    streaming_config,
                )

            elif event.type == "message_stop" and StreamingEventType.MESSAGE_STOP in streaming_config.event_types:
                await self._publish_event(
                    {
                        "type": "message_stop",
                        "message": {
                            "id": getattr(event.message, "id", None),
                            "role": getattr(event.message, "role", None),
                        },
                    },
                    streaming_config,
                )

            elif event.type == "input_json" and StreamingEventType.INPUT_JSON in streaming_config.event_types:
                await self._publish_event(
                    {
                        "type": "input_json",
                        "partial_json": event.partial_json,
                        "snapshot": event.snapshot,
                    },
                    streaming_config,
                )

            elif event.type == "tool_use" and StreamingEventType.TOOL_USE in streaming_config.event_types:
                await self._publish_event(
                    {
                        "type": "tool_use",
                        "tool_use": {
                            "id": getattr(event.tool_use, "id", None),  # Corrected attribute access
                            "name": getattr(event.tool_use, "name", None),  # Corrected attribute access
                            "input": getattr(event.tool_use, "input", {}),  # Corrected attribute access
                        },
                    },
                    streaming_config,
                )

        # Get accumulated message
        full_response = await stream.get_final_message()

        # If message_start data wasn't overwritten by final message, merge it
        if hasattr(full_response, "id") and not full_response.id and "id" in full_response_metadata:
            full_response.id = full_response_metadata["id"]
        if hasattr(full_response, "model") and not full_response.model and "model" in full_response_metadata:
            full_response.model = full_response_metadata["model"]
        if hasattr(full_response, "role") and not full_response.role and "role" in full_response_metadata:
            full_response.role = full_response_metadata["role"]
        if hasattr(full_response, "type") and not full_response.type and "type" in full_response_metadata:
            full_response.type = full_response_metadata["type"]
        if hasattr(full_response, "usage") and not full_response.usage and "usage" in full_response_metadata:
            # Attempt to reconstruct Usage object if possible, otherwise keep dict
            try:
                from anthropic.types import Usage  # Local import

                full_response.usage = Usage(**full_response_metadata["usage"])  # type: ignore
            except (ImportError, TypeError):
                full_response.usage = full_response_metadata["usage"]  # Keep as dict if Usage class unavailable/fails

        # Extract content text
        content = ""
        if hasattr(full_response, "content") and full_response.content:
            if isinstance(full_response.content, list):
                content = "".join(
                    block.text if hasattr(block, "text") else str(block)
                    for block in full_response.content
                    if getattr(block, "type", None) != "tool_use"
                )
            else:
                content = full_response.content

        return content, full_response

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        streaming_config: Optional[StreamingConfig] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Generate a response using Anthropic's API."""
        if self.client is None or self.async_client is None:
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

        api_kwargs = {**kwargs}
        if tools:
            api_kwargs["tools"] = tools

        api_kwargs["tool_choice"] = {
            "type": "auto",
        }

        if system_content:
            api_kwargs["system"] = system_content

        # Ensure a model is specified - use Claude 3 by default for tool calling
        if "model" not in api_kwargs:
            api_kwargs["model"] = "claude-3-7-sonnet-latest"

        # Ensure max_tokens is set
        if "max_tokens" not in api_kwargs:
            api_kwargs["max_tokens"] = 4096

        # Handle streaming if enabled
        if streaming_config and streaming_config.enabled:
            # Type ignoring arg-type error because mypy
            # can't verify that our message format matches Anthropic's expected format
            async with self.async_client.messages.stream(
                messages=anthropic_messages,  # type: ignore[arg-type]
                **api_kwargs,
            ) as stream:
                return await self._handle_stream(stream, streaming_config)

        # Non-streaming path (existing implementation)
        # Use a more direct approach that mypy can handle better
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(
                messages=anthropic_messages,  # type: ignore[arg-type]
                **api_kwargs,
            ),
        )

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
        response_objects = []

        if hasattr(response, "content") and isinstance(response.content, list):
            for block in response.content:
                response_objects.append(block.model_dump())
                if getattr(block, "type", None) == "tool_use":
                    tool_call = {
                        "id": getattr(block, "id", f"tool_{int(time.time())}"),
                        "name": getattr(block, "name", ""),
                        "arguments": getattr(block, "input", {}),
                    }
                    tool_calls.append(tool_call)

        tool_message = LLMMessage(
            role="assistant",
            content=response_objects,
            tool_calls=tool_calls,
        )

        return tool_calls, tool_message


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
