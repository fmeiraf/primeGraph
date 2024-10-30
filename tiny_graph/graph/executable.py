from typing import Dict

from tiny_graph.Buffer.last_value import Buffer
from tiny_graph.graph.base import Graph


class ExecutableGraph(Graph):
    def __init__(self):
        super().__init__()
        self.buffers: Dict[str, Buffer] = {
            field_name: Buffer(field_name, field_type)
            for field_name, field_type in self.state_schema.items()
        }

    def execute(self, execution_id: str) -> None:
        pass
