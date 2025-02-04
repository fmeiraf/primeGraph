from enum import Enum


class ChainStatus(Enum):
  IDLE = "IDLE"
  PAUSE = "PAUSE"
  RUNNING = "RUNNING"
  FAILED = "FAILED"
  DONE = "DONE"
