from enum import Enum, auto


class ChainStatus(Enum):
    IDLE = auto()
    PAUSE = auto()
    RUNNING = auto()
    FAILED = auto()
    ROUTING = auto()  # TODO: this is being used incorrectly and not preventing actions at it should. Should either improve or delete.
