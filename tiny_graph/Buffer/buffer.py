from typing import Any, Dict


class Buffer:
    def __init__(self, field_name: str, field_type: type):
        self.field_name = field_name
        self.field_type = field_type
        self.value: Any = None
        self.value_history: Dict[str, Any] = {}

    def update(self, new_value: Any, execution_id: str) -> None:
        if not isinstance(new_value, self.field_type):
            raise TypeError(
                f"Expected value of type {self.field_type}, got {type(new_value)}"
            )
        self.value = new_value
        self.add_history(new_value, execution_id)

    def get(self) -> Any:
        return self.value

    def add_history(self, value: Any, execution_id: str) -> None:
        self.value_history[execution_id] = value
