import hashlib
from typing import Any, Dict, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, model_validator

from tiny_graph.buffer.factory import BufferTypeMarker


class GraphState(BaseModel):
    """Base class for all graph states with buffer support"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    version: str = ""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name != "version":
            self.update_version()

    def update_version(self):
        """Update the version based on the current state of the model."""
        state = self.model_dump()
        state_str = str(sorted(state.items()))
        super().__setattr__("version", hashlib.md5(state_str.encode()).hexdigest())

    @model_validator(mode="before")
    @classmethod
    def wrap_buffer_types(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        hints = get_type_hints(cls)

        for field_name, field_type in hints.items():
            if field_name in values:
                origin = get_origin(field_type)
                if origin is not None and issubclass(origin, BufferTypeMarker):
                    buffer_instance = origin[field_type.__args__[0]](values[field_name])
                    values[field_name] = buffer_instance.initial_value

        return values

    @classmethod
    def get_buffer_types(cls) -> Dict[str, Any]:
        """Returns a mapping of field names to their buffer types"""
        # Use __annotations__ directly to avoid issues with get_type_hints

        annotations = cls.__annotations__
        buffer_types = {}

        for field_name, field_type in annotations.items():
            if issubclass(field_type, BufferTypeMarker):
                buffer_types[field_name] = field_type
            else:
                raise ValueError(
                    f"Field {field_name} is not using a buffer type (History, Incremental, LastValue, etc)"
                )

        return buffer_types
