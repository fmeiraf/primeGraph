import hashlib
from typing import Any, Dict, Type, TypeVar, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, model_validator

from primeGraph.buffer.factory import BufferTypeMarker, History


class GraphState(BaseModel):
    """Base class for all graph states with buffer support"""

    model_config = ConfigDict(
        arbitrary_types_allowed=False, strict=True, validate_default=True
    )
    version: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.update_version()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name != "version":
            self.update_version()

    def update_version(self):
        """Update the version based only on the model's attribute names."""
        # Get only the field names, ignoring their values
        field_names = sorted(
            [
                field_name
                for field_name in self.model_fields.keys()
                if field_name != "version"
            ]
        )
        state_str = str(field_names)
        super().__setattr__("version", hashlib.md5(state_str.encode()).hexdigest())

    @model_validator(mode="before")
    @classmethod
    def wrap_buffer_types(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        hints = get_type_hints(cls)

        def check_dict_type(value: Any, key_type: Type, value_type: Type) -> bool:
            if not isinstance(value, dict):
                return False
            return all(
                isinstance(k, key_type) and isinstance(v, value_type)
                for k, v in value.items()
            )

        for field_name, field_type in hints.items():
            if field_name in values:
                origin = get_origin(field_type)
                if origin is not None and issubclass(origin, BufferTypeMarker):
                    inner_type = field_type.__args__[0]
                    inner_origin = get_origin(inner_type)

                    # For History, ensure it's a list
                    if origin is History and not isinstance(values[field_name], list):
                        raise TypeError(f"Field {field_name} must be a list")

                    # For History with Dict type
                    if origin is History and inner_origin is dict:
                        key_type, value_type = get_args(inner_type)
                        if not all(
                            check_dict_type(item, key_type, value_type)
                            for item in values[field_name]
                        ):
                            raise TypeError(
                                f"All items in {field_name} must be Dict[{key_type.__name__}, {value_type.__name__}]"
                            )
                    # For History with List type
                    elif origin is History and inner_origin is list:
                        if not all(
                            isinstance(item, list) for item in values[field_name]
                        ):
                            raise TypeError(f"All items in {field_name} must be lists")
                    # For History with simple types
                    elif origin is History:
                        if not all(
                            isinstance(item, inner_type) for item in values[field_name]
                        ):
                            raise TypeError(
                                f"All items in {field_name} must be of type {inner_type}"
                            )
                    # For other buffer types with Dict
                    elif inner_origin is dict:
                        if not isinstance(values[field_name], dict):
                            raise TypeError(f"Field {field_name} must be a dict")
                    # For other buffer types with List
                    elif inner_origin is list:
                        if not isinstance(values[field_name], list):
                            raise TypeError(f"Field {field_name} must be a list")
                    # For other buffer types with simple types
                    elif inner_origin is None and not isinstance(inner_type, TypeVar):
                        if not isinstance(values[field_name], inner_type):
                            raise TypeError(
                                f"Field {field_name} must be of type {inner_type}"
                            )

        return values

    @classmethod
    def get_buffer_types(cls) -> Dict[str, Any]:
        """Returns a mapping of field names to their buffer types"""
        annotations = get_type_hints(cls)
        buffer_types = {}

        for field_name, field_type in annotations.items():
            if field_name == "version":
                continue

            origin = get_origin(field_type)
            if origin is not None and issubclass(origin, BufferTypeMarker):
                buffer_types[field_name] = field_type
            else:
                raise ValueError(
                    f"Field {field_name} is not using a buffer type (History, Incremental, LastValue, etc)"
                )

        return buffer_types
