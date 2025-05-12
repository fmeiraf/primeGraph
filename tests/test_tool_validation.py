from typing import Any, Dict, List, Optional

import pytest

from primeGraph.graph.tool_validation import (create_tool_args_model,
                                              validate_tool_args)


def test_create_tool_args_model_basic():
    """Test basic model creation with simple parameters."""
    def test_func(a: int, b: str, c: bool = True):
        pass

    model = create_tool_args_model(test_func)
    assert model.__name__ == "test_funcArgs"
    
    # Test valid data
    data = {"a": 1, "b": "test", "c": False}
    instance = model(**data)
    assert instance.a == 1
    assert instance.b == "test"
    assert instance.c is False

    # Test invalid data
    with pytest.raises(ValueError):
        model(a="not_an_int", b="test")


def test_create_tool_args_model_with_hidden_params():
    """Test model creation with hidden parameters."""
    def test_func(a: int, b: str, secret: str):
        pass

    model = create_tool_args_model(test_func, hidden_params=["secret"])
    assert model.__name__ == "test_funcArgs"
    
    # Test that secret is not in the model fields
    assert "secret" not in model.model_fields


def test_validate_tool_args_basic():
    """Test basic argument validation."""
    def test_func(a: int, b: str, c: bool = True):
        pass

    # Test valid arguments
    args = {"a": 1, "b": "test", "c": False}
    validated = validate_tool_args(test_func, args)
    assert validated == args

    # Test invalid arguments
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {"a": "not_an_int", "b": "test"})


def test_validate_tool_args_with_hidden_params():
    """Test argument validation with hidden parameters."""
    def test_func(a: int, b: str, secret: str):
        pass

    # Add tool definition with hidden params
    test_func._tool_definition = type("ToolDefinition", (), {"hidden_params": ["secret"]})()

    # Test with valid arguments including hidden param
    args = {"a": 1, "b": "test", "secret": "hidden_value"}
    validated = validate_tool_args(test_func, args)
    assert validated == args

    # Test that hidden param is passed through even with invalid type
    args = {"a": 1, "b": "test", "secret": 123}  # secret should be str but passes through
    validated = validate_tool_args(test_func, args)
    assert validated == args

    # Test that non-hidden params are still validated
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {"a": "not_an_int", "b": "test", "secret": "hidden"})


def test_validate_tool_args_with_complex_types():
    """Test argument validation with complex types."""
    class CustomType:
        def __init__(self, value: str):
            self.value = value

    def test_func(a: List[int], b: Dict[str, Any], c: CustomType):
        pass

    # Test with valid complex arguments
    args = {
        "a": [1, 2, 3],
        "b": {"key": "value"},
        "c": CustomType("test")
    }
    validated = validate_tool_args(test_func, args)
    assert validated == args

    # Test with invalid complex arguments
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {
            "a": "not_a_list",
            "b": "not_a_dict",
            "c": "not_a_custom_type"
        })


def test_validate_tool_args_with_optional_params():
    """Test argument validation with optional parameters."""
    def test_func(a: int, b: Optional[str] = None, c: List[int] = None):
        pass

    # Test with minimal required arguments
    args = {"a": 1}
    validated = validate_tool_args(test_func, args)
    assert validated == args

    # Test with all arguments
    args = {"a": 1, "b": "test", "c": [1, 2, 3]}
    validated = validate_tool_args(test_func, args)
    assert validated == args


def test_validate_tool_args_with_state_param():
    """Test argument validation with state parameter."""
    def test_func(state: Any, a: int, b: str):
        pass

    # Test that state parameter is ignored in validation
    args = {"a": 1, "b": "test"}
    validated = validate_tool_args(test_func, args)
    assert validated == args

    # Test that state parameter is not required
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {"b": "test"})  # Missing required 'a' parameter 


def test_validate_tool_args_with_dict_params():
    """Test argument validation with dictionary parameters."""
    def test_func(
        simple_dict: Dict[str, str],
        nested_dict: Dict[str, Dict[str, int]],
        mixed_dict: Dict[str, Any],
        optional_dict: Optional[Dict[str, str]] = None
    ):
        pass

    # Test with valid dictionary arguments
    args = {
        "simple_dict": {"key1": "value1", "key2": "value2"},
        "nested_dict": {"outer": {"inner1": 1, "inner2": 2}},
        "mixed_dict": {"str": "value", "int": 42, "list": [1, 2, 3]}
    }
    validated = validate_tool_args(test_func, args)
    assert validated == args
    assert isinstance(validated["simple_dict"], dict)
    assert isinstance(validated["nested_dict"], dict)
    assert isinstance(validated["mixed_dict"], dict)

    # Test with optional dictionary
    args_with_optional = {
        "simple_dict": {"key": "value"},
        "nested_dict": {"outer": {"inner": 1}},
        "mixed_dict": {"key": "value"},
        "optional_dict": {"opt_key": "opt_value"}
    }
    validated = validate_tool_args(test_func, args_with_optional)
    assert validated == args_with_optional
    assert validated["optional_dict"] == {"opt_key": "opt_value"}

    # Test with invalid dictionary types
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {
            "simple_dict": "not_a_dict",
            "nested_dict": {"outer": "not_a_dict"},
            "mixed_dict": {"key": "value"}
        })

    # Test with invalid nested dictionary structure
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {
            "simple_dict": {"key": "value"},
            "nested_dict": {"outer": "not_a_dict"},
            "mixed_dict": {"key": "value"}
        })

    # Test with missing required dictionary
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {
            "nested_dict": {"outer": {"inner": 1}},
            "mixed_dict": {"key": "value"}
        })

    # Test with empty dictionaries
    args_empty = {
        "simple_dict": {},
        "nested_dict": {"outer": {}},
        "mixed_dict": {}
    }
    validated = validate_tool_args(test_func, args_empty)
    assert validated == args_empty

    # Test with None for optional dictionary
    args_none_optional = {
        "simple_dict": {"key": "value"},
        "nested_dict": {"outer": {"inner": 1}},
        "mixed_dict": {"key": "value"},
        "optional_dict": None
    }
    validated = validate_tool_args(test_func, args_none_optional)
    assert validated == args_none_optional
    assert validated["optional_dict"] is None 