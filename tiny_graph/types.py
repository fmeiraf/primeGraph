from enum import Enum, auto


class ChainStatus(Enum):
    IDLE = auto()
    PAUSE = auto()
    RUNNING = auto()
    FAILED = auto()
