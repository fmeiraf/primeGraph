from typing import Any

from tiny_graph.buffer.base import BaseBuffer


class LastValueBuffer(BaseBuffer):
    """Buffer that stores the last value of a field."""

    def __init__(self, field_name: str, field_type: type):
        super().__init__(field_name, field_type)

    def update(self, new_value: Any, execution_id: str) -> None:
        with self._lock:
            self._enforce_type(new_value)

            self.value = new_value
            self.last_value = new_value
            self.add_history(new_value, execution_id)
            self._consumed = False

    def get(self) -> Any:
        with self._lock:
            return self.value
