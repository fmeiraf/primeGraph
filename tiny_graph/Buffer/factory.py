from typing import Any, Generic, Type, TypeVar, get_args, get_origin

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from tiny_graph.buffer.base import BaseBuffer
from tiny_graph.buffer.history import HistoryBuffer
from tiny_graph.buffer.incremental import IncrementalBuffer
from tiny_graph.buffer.last_value import LastValueBuffer

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

    def __class_getitem__(cls, item):
        return type(f"{cls.__name__}[{item.__name__}]", (cls,), {"_inner_type": item})

    @property
    def inner_type(self):
        return self._inner_type

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

        return handler.generate_schema(args[0])


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

        return core_schema.list_schema(handler.generate_schema(args[0]))


class Incremental(BufferTypeMarker[T]):
    pass


class LastValue(BufferTypeMarker[T]):
    pass


# Buffer Factory
class BufferFactory:
    @staticmethod
    def create_buffer(field_name: str, annotation: Type) -> BaseBuffer:
        buffer_type = annotation.__bases__[0].__name__

        buffer_map = {
            "History": HistoryBuffer,
            "Incremental": IncrementalBuffer,
            "LastValue": LastValueBuffer,
        }

        buffer_type = buffer_map.get(buffer_type, LastValueBuffer)
        actual_type = annotation._inner_type
        buffer = buffer_type(field_name, actual_type)

        # Set initial value if available
        if hasattr(annotation, "initial_value"):
            initial_value = annotation.initial_value
            if buffer_type == HistoryBuffer:
                if initial_value and not isinstance(initial_value, list):
                    raise TypeError(
                        f"HistoryBuffer initial value must be a list, got {type(initial_value)}"
                    )
            buffer.value = initial_value

        return buffer
