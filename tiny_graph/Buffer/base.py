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

    def _enforce_type(self, new_value: Any) -> None:
        actual_type = self.field_type
        if hasattr(actual_type, "_inner_type"):
            actual_type = actual_type._inner_type

        # Handle generic types like Dict, List, etc.
        origin = get_origin(actual_type)
        if origin is not None:
            # Check if the value matches the generic container type (dict, list, etc)
            if not isinstance(new_value, origin):
                raise TypeError(
                    f"Expected value of type {origin.__name__}, got {type(new_value).__name__}"
                )

            # Get the type arguments (e.g., str and float for Dict[str, float])
            type_args = get_args(actual_type)

            if origin is dict:
                # Check key and value types for dictionaries
                for key, value in new_value.items():
                    if not isinstance(key, type_args[0]):
                        raise TypeError(
                            f"Dict key must be {type_args[0].__name__}, got {type(key).__name__}"
                        )
                    if not isinstance(value, type_args[1]):
                        raise TypeError(
                            f"Dict value must be {type_args[1].__name__}, got {type(value).__name__}"
                        )
            elif origin is list:
                # Check element types for lists
                for item in new_value:
                    if not isinstance(item, type_args[0]):
                        raise TypeError(
                            f"List items must be {type_args[0].__name__}, got {type(item).__name__}"
                        )
        else:
            # Regular non-generic type
            if not isinstance(new_value, actual_type):
                raise TypeError(
                    f"Expected value of type {actual_type.__name__}, got {type(new_value).__name__}"
                )
