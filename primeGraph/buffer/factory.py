from typing import Any, Dict, Generic, Type, TypeVar, get_args, get_origin

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from primeGraph.buffer.base import BaseBuffer
from primeGraph.buffer.history import HistoryBuffer
from primeGraph.buffer.incremental import IncrementalBuffer
from primeGraph.buffer.last_value import LastValueBuffer

T = TypeVar("T")


# Buffer Type Markers
class BufferTypeMarker(Generic[T]):
  def __init__(self, initial_value: Any = None):
    self.initial_value = initial_value
    self._inner_type = None

    if hasattr(self, "__orig_class__"):
      self._inner_type = get_args(self.__orig_class__)[0]
    elif hasattr(self.__class__, "_inner_type"):
      self._inner_type = self.__class__._inner_type

  @classmethod
  def __get_pydantic_core_schema__(
    cls,
    source_type: Any,
    handler: GetCoreSchemaHandler,
  ) -> CoreSchema:
    origin = get_origin(source_type)
    if origin is None:
      return core_schema.any_schema()

    args = get_args(source_type)
    if not args:
      return core_schema.any_schema()

    inner_schema = handler.generate_schema(args[0])
    return inner_schema


class History(BufferTypeMarker[T]):
  @classmethod
  def __get_pydantic_core_schema__(
    cls,
    source_type: Any,
    handler: GetCoreSchemaHandler,
  ) -> CoreSchema:
    origin = get_origin(source_type)
    if origin is None:
      return core_schema.list_schema(core_schema.any_schema())

    args = get_args(source_type)
    if not args:
      return core_schema.list_schema(core_schema.any_schema())

    inner_schema = handler.generate_schema(args[0])
    return core_schema.list_schema(items_schema=inner_schema, strict=True)


class Incremental(BufferTypeMarker[T]):
  pass


# TODO: this is not working with typing.Dict
class LastValue(BufferTypeMarker[T]):
  @classmethod
  def __get_pydantic_core_schema__(
    cls,
    source_type: Any,
    handler: GetCoreSchemaHandler,
  ) -> CoreSchema:
    origin = get_origin(source_type)
    if origin is None:
      return core_schema.any_schema()

    args = get_args(source_type)
    if not args:
      return core_schema.any_schema()

    inner_schema = handler.generate_schema(args[0])
    return inner_schema


# Buffer Factory
class BufferFactory:
  @staticmethod
  def create_buffer(field_name: str, annotation: Type) -> BaseBuffer:
    # Get the origin type for generic types
    origin = get_origin(annotation) or annotation

    # Map buffer types to their implementations
    buffer_map = {
      History: HistoryBuffer,
      Incremental: IncrementalBuffer,
      LastValue: LastValueBuffer,
    }

    buffer_type = buffer_map.get(origin, LastValueBuffer)

    # Get the inner type from the generic's args
    inner_type = get_args(annotation)[0] if get_args(annotation) else Any

    buffer = buffer_type(field_name, inner_type)  # type: ignore

    # Set initial value if available
    if hasattr(annotation, "initial_value"):
      initial_value = annotation.initial_value
      if buffer_type == HistoryBuffer and initial_value and not isinstance(initial_value, list):
        raise TypeError(f"HistoryBuffer initial value must be a list, got {type(initial_value)}")
      buffer.value = initial_value

    return buffer

  @classmethod
  def wrap_buffer_types(cls, values: Dict[str, Any], model_fields: Dict[str, Any]) -> Dict[str, Any]:
    for field_name, field in model_fields.items():
      if field_name in values:
        origin_type = get_origin(field.annotation)
        if origin_type in (History, Incremental, LastValue):
          inner_type = get_args(field.annotation)[0]
          inner_origin = get_origin(inner_type)

          # Get the value to check
          value = values[field_name]

          # Handle different type scenarios
          if inner_origin is list:
            if not isinstance(value, list):
              raise TypeError(f"Field {field_name} must be a list")
            # Optionally check list contents
            list_type_args = get_args(inner_type)
            if list_type_args:
              # Skip validation of list contents - let Pydantic handle it
              pass
          elif inner_origin is dict:
            if not isinstance(value, dict):
              raise TypeError(f"Field {field_name} must be a dict")
            # Optionally check dict contents
            dict_type_args = get_args(inner_type)
            if dict_type_args:
              # Skip validation of dict contents - let Pydantic handle it
              pass
          elif inner_origin is None and not isinstance(inner_type, TypeVar):
            # Only check simple, non-generic types
            if not isinstance(value, inner_type):
              raise TypeError(f"Field {field_name} must be of type {inner_type}")
    return values
