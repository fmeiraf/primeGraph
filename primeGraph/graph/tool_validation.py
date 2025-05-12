import inspect
from typing import Any, Callable, Dict, List, Optional, Type, cast

from pydantic import BaseModel, ConfigDict, Field, create_model


class ToolDefinition:
    """Type for tool definition to help with type checking."""

    hidden_params: List[str]


def create_tool_args_model(func: Callable[..., Any], hidden_params: Optional[List[str]] = None) -> Type[BaseModel]:
    """
    Create a Pydantic model for validating tool arguments based on the function's signature.

    Args:
        func: The tool function to create a validation model for
        hidden_params: List of parameter names to skip validation for

    Returns:
        A Pydantic model class for validating the tool's arguments
    """
    sig = inspect.signature(func)
    fields: Dict[str, tuple[Any, Any]] = {}
    hidden_params = hidden_params or []

    for name, param in sig.parameters.items():
        if name == "state" or name in hidden_params:  # Skip state and hidden parameters
            continue

        # Get the parameter type
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any

        # Handle default values
        if param.default != inspect.Parameter.empty:
            fields[name] = (param_type, Field(default=param.default))
        else:
            fields[name] = (param_type, Field())

    # Create the model with strict validation
    model = create_model(
        f"{func.__name__}Args", __config__=ConfigDict(strict=True, arbitrary_types_allowed=True), **fields
    )  # type: ignore
    return cast(Type[BaseModel], model)


def validate_tool_args(
    func: Callable[..., Any], arguments: Dict[str, Any], hidden_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate tool arguments against the function's signature using Pydantic.
    Hidden parameters are passed through without validation.

    Args:
        func: The tool function to validate arguments for
        arguments: The arguments to validate
        hidden_params: List of parameter names to skip validation for

    Returns:
        The validated arguments as a dictionary, with hidden parameters included

    Raises:
        ValueError: If arguments fail validation
    """
    # Get hidden parameters from tool definition if available
    if hasattr(func, "_tool_definition"):
        tool_def = cast(ToolDefinition, getattr(func, "_tool_definition"))
        if hasattr(tool_def, "hidden_params"):
            hidden_params = tool_def.hidden_params

    # Create validation model
    ArgsModel = create_tool_args_model(func, hidden_params)

    try:
        # Extract hidden parameters
        hidden_args = {k: v for k, v in arguments.items() if k in (hidden_params or [])}

        # Get only the arguments that were provided (excluding hidden params)
        provided_args = {k: v for k, v in arguments.items() if k not in (hidden_params or [])}

        # Validate non-hidden arguments
        validated = ArgsModel(**provided_args)

        # Get only the fields that were provided in the input
        result = {k: v for k, v in validated.model_dump().items() if k in provided_args}

        # Add hidden arguments back
        result.update(hidden_args)

        return result
    except Exception as e:
        raise ValueError(f"Invalid arguments for tool {func.__name__}: {str(e)}")
