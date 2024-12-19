from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, get_args, get_origin


class BaseBuffer(ABC):
  """Buffer base class.

  Buffers are used to store the state of a field across executions.
  This helps isolating the different parts of the state, making updates easier and quicker during concurrent executions.
  """

  def __init__(self, field_name: str, field_type: type):
    self.field_name = field_name
    self.field_type = field_type
    self.value: Any = None
    self.last_value: Any = None
    self.value_history: Dict[str, Any] = {}
    self._lock = Lock()
    self._ready_for_consumption = False
    self._has_state = False

  @abstractmethod
  def update(self, new_value: Any, execution_id: str) -> None:
    pass

  @abstractmethod
  def get(self) -> Any:
    pass

  @abstractmethod
  def set_value(self, value: Any) -> None:
    pass

  def add_history(self, value: Any, execution_id: str) -> None:
    self.value_history[execution_id] = value

  def consume_last_value(self) -> Any:
    with self._lock:
      if isinstance(self.last_value, list):
        last_value_copy = self.last_value.copy()
        self.last_value = []
      else:
        last_value_copy = self.last_value
        self.last_value = None
      self._ready_for_consumption = False
    return last_value_copy

  def _enforce_type(self, new_value: Any) -> None:
    """Enforce the type of the buffer value."""
    if self.field_type is None or new_value is None:
      return

    def validate_dict_contents(value: dict, key_type: type, value_type: type) -> None:
      for k, v in value.items():
        if not isinstance(k, key_type):
          raise TypeError(f"Dict key must be {key_type}, got {type(k)}")
        validate_value(v, value_type)

    def validate_list_contents(value: list, item_type: type) -> None:
      for item in value:
        validate_value(item, item_type)

    def validate_value(value: Any, expected_type: type) -> None:
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

    validate_value(new_value, self.field_type)
