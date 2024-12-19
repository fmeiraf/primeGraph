import hashlib
from typing import Any, Dict, Type, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, model_validator

from primeGraph.buffer.factory import BufferTypeMarker, History


class GraphState(BaseModel):
  """Base class for all graph states with buffer support"""

  model_config = ConfigDict(arbitrary_types_allowed=False, strict=True, validate_default=True)
  version: str = ""

  def __init__(self, **data: Any) -> None:
    super().__init__(**data)
    self.update_version()

  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
    if name != "version":
      self.update_version()

  def update_version(self) -> None:
    """Update the version based only on the model's attribute names."""
    # Get only the field names, ignoring their values
    field_names = sorted([field_name for field_name in self.model_fields if field_name != "version"])
    state_str = str(field_names)
    super().__setattr__("version", hashlib.md5(state_str.encode()).hexdigest())

  @model_validator(mode="before")
  @classmethod
  def wrap_buffer_types(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    hints = get_type_hints(cls)

    def validate_dict_contents(value: dict, key_type: Type, value_type: Type) -> None:
      for k, v in value.items():
        if not isinstance(k, key_type):
          raise TypeError(f"Dict key must be {key_type}, got {type(k)}")
        validate_value(v, value_type)

    def validate_list_contents(value: list, item_type: Type) -> None:
      for item in value:
        validate_value(item, item_type)

    def validate_value(value: Any, expected_type: Type) -> None:
      # Special case: Any accepts everything
      if expected_type is Any:
        return

      origin = get_origin(expected_type)
      if origin is None:
        if not isinstance(value, expected_type):
          raise TypeError(f"Value must be {expected_type}, got {type(value)}")
        return

      if origin is dict:
        if not isinstance(value, dict):
          raise TypeError(f"Value must be dict, got {type(value)}")
        key_type, value_type = get_args(expected_type)
        validate_dict_contents(value, key_type, value_type)
      elif origin is list:
        if not isinstance(value, list):
          raise TypeError(f"Value must be list, got {type(value)}")
        item_type = get_args(expected_type)[0]
        validate_list_contents(value, item_type)

    for field_name, field_type in hints.items():
      if field_name not in values:
        continue

      origin = get_origin(field_type)
      if origin is not None and issubclass(origin, BufferTypeMarker):
        inner_type = get_args(field_type)[0]
        value = values[field_name]

        if origin is History:
          if not isinstance(value, list):
            raise TypeError(f"Field {field_name} must be a list")
          for item in value:
            validate_value(item, inner_type)
        else:
          validate_value(value, inner_type)

    return values

  @classmethod
  def get_buffer_types(cls) -> Dict[str, Any]:
    """Returns a mapping of field names to their buffer types"""
    annotations = get_type_hints(cls)
    buffer_types = {}

    for field_name, field_type in annotations.items():
      if field_name == "version":
        continue

      origin = get_origin(field_type)
      if origin is not None and issubclass(origin, BufferTypeMarker):
        buffer_types[field_name] = field_type
      else:
        raise ValueError(f"Field {field_name} is not using a buffer type (History, Incremental, LastValue, etc)")

    return buffer_types
