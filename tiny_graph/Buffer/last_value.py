from typing import Any

from .base_buffer import BaseBuffer


class LastValueBuffer(BaseBuffer):
    """Buffer that stores the last value of a field."""

    def __init__(self, field_name: str, field_type: type):
        super().__init__(field_name, field_type)

    def update(self, new_value: Any, execution_id: str) -> None:
        with self._lock:
            if not isinstance(new_value, self.field_type):
                raise TypeError(
                    f"Expected value of type {self.field_type}, got {type(new_value)}"
                )
            self.value = new_value
            self.last_value = new_value
            self.add_history(new_value, execution_id)

    def get(self) -> Any:
        with self._lock:
            return self.value
