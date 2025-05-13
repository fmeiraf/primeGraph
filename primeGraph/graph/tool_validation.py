import ast
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model


class ToolDefinition:
    """Type for tool definition to help with type checking."""

    hidden_params: List[str]


def _parse_value_to_type(value: Any, expected_type: Type) -> Any:  # type: ignore
    """
    Parse a value to its expected type, handling string serialization from LLM responses.

    Args:
        value: The value to parse
        expected_type: The expected type to parse into

    Returns:
        The parsed value of the expected type
    """
    # Handle None values first
    if value is None:
        origin_type = get_origin(expected_type)
        if origin_type is Union:
            types = get_args(expected_type)
            if type(None) in types:
                return None
        elif expected_type is type(None):
            return None
        raise ValueError(f"Value None is not valid for type {expected_type}")

    # Handle Any type - return the value as is
    if expected_type is Any:
        return value

    # Handle object type - return the value as is
    if expected_type is object:
        return value

    # Unwrap double-quoted JSON strings recursively for primitives
    if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
        return _parse_value_to_type(value[1:-1], expected_type)

    origin_type = get_origin(expected_type)
    if origin_type is not None:
        if origin_type is Union:
            types = get_args(expected_type)
            types = tuple(t for t in types if t is not type(None))
            for t in types:
                try:
                    return _parse_value_to_type(value, t)
                except (ValueError, TypeError):
                    continue
            raise ValueError(f"Value {value} does not match any of the expected types: {types}")

        if origin_type is list:
            if isinstance(value, str):
                print(f"DEBUG: Parsing list value: {repr(value)} (expected_type={expected_type})")
                try:
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    value = json.loads(value)
                except json.JSONDecodeError:
                    # Try ast.literal_eval as a second resort for Python-style lists
                    try:
                        value = ast.literal_eval(value)
                    except Exception:
                        # If the string looks like a list, raise an error
                        if value.strip().startswith("[") and value.strip().endswith("]"):
                            raise ValueError(f"Could not parse string as list: {value}")
                        # Only fallback to comma-split if the string does NOT look like a JSON array
                        value = [v.strip() for v in value.split(",") if v.strip()]
                if not isinstance(value, list):
                    raise ValueError(f"Expected list, got {type(value)}")
                inner_type_raw = get_args(expected_type)[0] if get_args(expected_type) else Any
                inner_type = inner_type_raw if isinstance(inner_type_raw, type) else Any
                return [_parse_value_to_type(item, cast(type, inner_type)) for item in value]

        elif origin_type is dict:
            if isinstance(value, str):
                try:
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    value = json.loads(value)
                except json.JSONDecodeError:
                    try:
                        value = value.replace("'", '"')
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        raise ValueError(f"Could not parse string as dict: {value}")
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict, got {type(value)}")
            args = get_args(expected_type)
            key_type_raw = args[0] if len(args) > 0 else object
            value_type_raw = args[1] if len(args) > 1 else object
            key_type: type = key_type_raw if isinstance(key_type_raw, type) and key_type_raw is not None else object
            value_type: type = (
                value_type_raw if isinstance(value_type_raw, type) and value_type_raw is not None else object
            )
            dict_value = cast(
                Dict[Any, Any],
                {_parse_value_to_type(k, key_type): _parse_value_to_type(v, value_type) for k, v in value.items()},
            )  # type: ignore[arg-type]
            return dict_value

    # Special handling for plain dict (not typing.Dict)
    if expected_type is dict:
        if isinstance(value, str):
            try:
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    value = value.replace("'", '"')
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse string as dict: {value}")
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")
        return value

    # Special handling for plain list (not typing.List)
    if expected_type is list:
        if isinstance(value, str):
            try:
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    if value.strip().startswith("[") and value.strip().endswith("]"):
                        raise ValueError(f"Could not parse string as list: {value}")
                    value = [v.strip() for v in value.split(",") if v.strip()]
            if not isinstance(value, list):
                raise ValueError(f"Expected list, got {type(value)}")
        return value

    if expected_type is bool:
        if isinstance(value, str):
            value = value.lower()
            if value in ("true", "1", "yes", "y"):
                return True
            if value in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"Could not parse '{value}' as boolean")
        return bool(value)

    if expected_type is int:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    float_val = float(value)
                    if float_val.is_integer():
                        return int(float_val)
                    else:
                        raise ValueError(f"Could not parse '{value}' as integer (not an integer value)")
                except ValueError:
                    raise ValueError(f"Could not parse '{value}' as integer")
        return int(value)

    if expected_type is float:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Could not parse '{value}' as float")
        return float(value)

    if expected_type is str:
        return str(value)

    # For custom types, check if the value is already of the expected type
    # by comparing the actual type with the expected type
    if type(value).__name__ == expected_type.__name__:
        return value

    try:
        return expected_type(value)
    except (ValueError, TypeError):
        if hasattr(expected_type, "__module__") and expected_type.__module__ != "builtins":
            return value
        raise ValueError(f"Could not convert {value} to {expected_type}")


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

    # Get type hints for better type information
    type_hints = get_type_hints(func)

    for name, param in sig.parameters.items():
        if name == "state" or name in hidden_params:  # Skip state and hidden parameters
            continue

        # Get the parameter type, preferring type hints over annotations
        param_type = type_hints.get(name, param.annotation if param.annotation != inspect.Parameter.empty else Any)

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
    Handles string-serialized arguments from LLM responses.

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

        # Parse string-serialized arguments to their expected types
        parsed_args = {}
        for name, value in provided_args.items():
            if name in ArgsModel.model_fields:
                expected_type = ArgsModel.model_fields[name].annotation
                parsed_args[name] = _parse_value_to_type(value, expected_type)  # type: ignore
            else:
                parsed_args[name] = value

        # Validate non-hidden arguments
        validated = ArgsModel(**parsed_args)

        # Get only the fields that were provided in the input
        result = {k: v for k, v in validated.model_dump().items() if k in provided_args}

        # Add hidden arguments back
        result.update(hidden_args)

        return result
    except Exception as e:
        raise ValueError(f"Invalid arguments for tool {func.__name__}: {str(e)}")
