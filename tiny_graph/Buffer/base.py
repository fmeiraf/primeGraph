from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict


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

    @abstractmethod
    def update(self, new_value: Any, execution_id: str) -> None:
        pass

    @abstractmethod
    def get(self) -> Any:
        pass

    def add_history(self, value: Any, execution_id: str) -> None:
        self.value_history[execution_id] = value

    def consume_last_value(self) -> Any:
        with self._lock:
            last_value = self.last_value
            self.last_value = None
            return last_value
