import inspect
from functools import wraps
from typing import Any, Callable, TypeVar

# Minimum stack depth needed for internal method check
MIN_STACK_DEPTH = 2

T = TypeVar("T")  # Define a generic type variable


def internal_only(func: Callable[..., T]) -> Callable[..., T]:
  @wraps(func)
  def wrapper(*args: Any, **kwargs: Any) -> T:
    # Get the call stack
    stack = inspect.stack()

    # We need at least the wrapper frame and the caller frame
    if len(stack) >= MIN_STACK_DEPTH:
      # Get the caller frame (the method that called our decorated method)
      caller_frame = stack[1]

      # Get the instance that's calling the method (self in the caller's context)
      caller_self = caller_frame.frame.f_locals.get("self", None)
      # Get the current instance (self in our decorated method)
      current_self = args[0] if args else None

      # Check if both instances exist and are the same class
      if (
        caller_self is not None
        and current_self is not None
        and isinstance(caller_self, current_self.__class__)
        and caller_self is current_self
      ):
        return func(*args, **kwargs)

    raise RuntimeError(
      f"Method {func.__name__} can only be called internally by other methods of {args[0].__class__.__name__}"
    )

  return wrapper
