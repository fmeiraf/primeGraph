from typing import Any, Union

from primeGraph.buffer.base import BaseBuffer


class IncrementalBuffer(BaseBuffer):
  """Buffer that stores the incremental value of a field."""

  def __init__(self, field_name: str, field_type: type):
    super().__init__(field_name, field_type)
    self.value: Union[int, float] = 0
    self.last_value: Union[int, float] = 0

  def update(self, new_value: Any, execution_id: str) -> None:
    with self._lock:
      self._enforce_type(new_value)
      self.value = self.value + new_value
      self.last_value = self.value
      self.add_history(self.value, execution_id)
      self._ready_for_consumption = True

  def get(self) -> Any:
    with self._lock:
      return self.value

  def set_value(self, value: Any) -> None:
    with self._lock:
      self.value = value
      self.last_value = value
