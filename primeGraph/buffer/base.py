from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, Union, get_args, get_origin


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
        # with self._lock:
        #     self.value_history[execution_id] = value
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

    def _enforce_type(self, new_value):
        """Enforce the type of the buffer value."""
        if self.field_type is None:
            return

        # Get the origin type (e.g., List from List[str])
        origin = get_origin(self.field_type) or self.field_type

        # Handle Union types specially
        if origin is Union:
            # Get the allowed types from the Union
            allowed_types = get_args(self.field_type)
            if not any(isinstance(new_value, t) for t in allowed_types):
                raise TypeError(
                    f"Buffer value must be one of types {allowed_types}, got {type(new_value)}"
                )
            return

        # For non-Union types, proceed with normal isinstance check
        if not isinstance(new_value, origin):
            if new_value:
                raise TypeError(
                    f"Buffer value must be of type {self.field_type}, got {type(new_value)}"
                )
