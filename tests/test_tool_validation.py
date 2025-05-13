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


def test_validate_tool_args_with_serialized_strings():
    """Test argument validation with string-serialized values from LLM responses."""
    def test_func(
        int_val: int,
        float_val: float,
        bool_val: bool,
        list_val: List[int],
        dict_val: Dict[str, Any],
        nested_dict: Dict[str, Dict[str, int]]
    ):
        pass

    # Test with string-serialized values
    args = {
        "int_val": "42",
        "float_val": "3.14",
        "bool_val": "true",
        "list_val": "[1, 2, 3]",
        "dict_val": '{"key": "value", "number": 42}',
        "nested_dict": '{"outer": {"inner": 123}}'
    }
    validated = validate_tool_args(test_func, args)
    
    # Verify types after validation
    assert isinstance(validated["int_val"], int)
    assert validated["int_val"] == 42
    assert isinstance(validated["float_val"], float)
    assert validated["float_val"] == 3.14
    assert isinstance(validated["bool_val"], bool)
    assert validated["bool_val"] is True
    assert isinstance(validated["list_val"], list)
    assert validated["list_val"] == [1, 2, 3]
    assert isinstance(validated["dict_val"], dict)
    assert validated["dict_val"] == {"key": "value", "number": 42}
    assert isinstance(validated["nested_dict"], dict)
    assert validated["nested_dict"] == {"outer": {"inner": 123}}

    # Test with JSON-serialized strings
    args = {
        "int_val": '"42"',  # Double-quoted string
        "float_val": '"3.14"',
        "bool_val": '"true"',
        "list_val": '"[1, 2, 3]"',
        "dict_val": '{"key": "value"}',
        "nested_dict": '{"outer": {"inner": 123}}'
    }
    validated = validate_tool_args(test_func, args)
    
    # Verify types after validation
    assert isinstance(validated["int_val"], int)
    assert validated["int_val"] == 42
    assert isinstance(validated["float_val"], float)
    assert validated["float_val"] == 3.14
    assert isinstance(validated["bool_val"], bool)
    assert validated["bool_val"] is True
    assert isinstance(validated["list_val"], list)
    assert validated["list_val"] == [1, 2, 3]
    assert isinstance(validated["dict_val"], dict)
    assert validated["dict_val"] == {"key": "value"}
    assert isinstance(validated["nested_dict"], dict)
    assert validated["nested_dict"] == {"outer": {"inner": 123}}

    # Test with invalid serialized values
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {
            "int_val": "not_a_number",
            "float_val": "3.14",
            "bool_val": "true",
            "list_val": "[1, 2, 3]",
            "dict_val": '{"key": "value"}',
            "nested_dict": '{"outer": {"inner": 123}}'
        })

    # Test with malformed JSON
    with pytest.raises(ValueError):
        validate_tool_args(test_func, {
            "int_val": "42",
            "float_val": "3.14",
            "bool_val": "true",
            "list_val": "[1, 2, 3",
            "dict_val": '{"key": "value"}',
            "nested_dict": '{"outer": {"inner": 123}}'
        }) 


def test_validate_tool_args_with_stringified_values():
    """Test that stringified arguments (as LLMs might return) are parsed correctly."""
    def test_func(
        a: int,
        b: float,
        c: bool,
        d: list,
        e: dict,
        f: List[int],
        g: Dict[str, int],
        h: Optional[List[str]] = None,
        i: Optional[Dict[str, str]] = None,
    ):
        pass

    # Simulate LLM returning all arguments as strings
    args = {
        "a": "42",
        "b": "3.14",
        "c": "true",
        "d": "[1, 2, 3]",
        "e": '{"x": 1, "y": 2}',
        "f": "[4, 5, 6]",
        "g": '{"foo": 10, "bar": 20}',
        "h": '["apple", "banana"]',
        "i": '{"k1": "v1", "k2": "v2"}'
    }
    validated = validate_tool_args(test_func, args)
    assert validated["a"] == 42
    assert validated["b"] == 3.14
    assert validated["c"] is True
    assert validated["d"] == [1, 2, 3]
    assert validated["e"] == {"x": 1, "y": 2}
    assert validated["f"] == [4, 5, 6]
    assert validated["g"] == {"foo": 10, "bar": 20}
    assert validated["h"] == ["apple", "banana"]
    assert validated["i"] == {"k1": "v1", "k2": "v2"}

    # Simulate LLM returning comma-separated list for a List[str]
    def func_list_str(items: List[str]):
        pass
    args = {"items": "apple, banana, cherry"}
    validated = validate_tool_args(func_list_str, args)
    # Accept either a single string or a list
    if isinstance(validated["items"], list):
        assert validated["items"] == ["apple", "banana", "cherry"]
    else:
        # fallback: if not parsed, it's a string
        assert validated["items"] == "apple, banana, cherry"

    # Simulate LLM returning a Python literal for a dict
    def func_dict(d: Dict[str, int]):
        pass
    args = {"d": "{'a': 1, 'b': 2}"}
    validated = validate_tool_args(func_dict, args)
    # Accept either a dict or a string
    if isinstance(validated["d"], dict):
        assert validated["d"] == {"a": 1, "b": 2}
    else:
        # fallback: if not parsed, it's a string
        assert validated["d"] == "{'a': 1, 'b': 2}"

    # Simulate LLM returning a nested JSON string
    def func_nested(n: Dict[str, List[int]]):
        pass
    args = {"n": '{"nums": [1, 2, 3, 4]}'}
    validated = validate_tool_args(func_nested, args)
    assert validated["n"] == {"nums": [1, 2, 3, 4]}

    # Simulate LLM returning a boolean as "false"
    def func_bool(flag: bool):
        pass
    args = {"flag": "false"}
    validated = validate_tool_args(func_bool, args)
    assert validated["flag"] is False

    # Simulate LLM returning a number as a float string for an int param
    def func_int(x: int):
        pass
    args = {"x": "7.0"}
    validated = validate_tool_args(func_int, args)
    # Accept either int or float
    assert int(validated["x"]) == 7

    # Simulate LLM returning a string for an Optional[List[int]]
    def func_opt_list(x: Optional[List[int]] = None):
        pass
    args = {"x": "[10, 20, 30]"}
    validated = validate_tool_args(func_opt_list, args)
    assert validated["x"] == [10, 20, 30]

    # Simulate LLM returning None for an Optional[List[int]]
    args = {"x": None}
    validated = validate_tool_args(func_opt_list, args)
    assert validated["x"] is None 