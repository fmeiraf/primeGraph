# checkpoint_library/serialization.py

from typing import Type

from pydantic import BaseModel


def serialize_model(model_instance: BaseModel) -> str:
  """
  Serialize a Pydantic model instance to a JSON string.
  """
  try:
    return model_instance.model_dump_json()
  except Exception as e:
    raise SerializationError(f"Failed to serialize model: {e}") from e


def deserialize_model(model_class: Type[BaseModel], data: str) -> BaseModel:
  """
  Deserialize a JSON string to a Pydantic model instance.
  """
  try:
    return model_class.model_validate_json(data)
  except Exception as e:
    raise DeserializationError(f"Failed to deserialize data: {e}") from e


class SerializationError(Exception):
  pass


class DeserializationError(Exception):
  pass
