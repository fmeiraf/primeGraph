from typing import Any, Callable, List, Literal, NamedTuple, Union

from pydantic import BaseModel

# from tiny_graph.Buffer.last_value import Buffer
from tiny_graph.graph.base import BaseGraph


class ExecutableNode(NamedTuple):
    node_name: str
    task_list: List[Callable]
    execution_type: Literal["sequential", "parallel"]


class Graph(BaseGraph):
    def __init__(self, state: Union[BaseModel, NamedTuple, None] = None):
        super().__init__(state)
        # self.buffers: Dict[str, Buffer] = {
        #     field_name: Buffer(field_name, field_type)
        #     for field_name, field_type in self.state_schema.items()
        # }

    def _convert_execution_plan(self) -> List[Any]:
        """Converts the execution plan to a list of functions that have concurrency flags"""

        self._force_compile()

        def create_executable_node(
            exec_plan_item: Union[str, List[Any], None],
        ) -> ExecutableNode:
            if isinstance(exec_plan_item, str):
                return ExecutableNode(
                    node_name=exec_plan_item,
                    task_list=[self.nodes[exec_plan_item].action],
                    execution_type="sequential",
                )
            elif isinstance(exec_plan_item, list):
                return ExecutableNode(
                    node_name=f"group_{exec_plan_item[0]}",
                    task_list=[create_executable_node(item) for item in exec_plan_item],
                    execution_type="parallel",
                )
            else:
                raise ValueError(f"Expected str or list, got {type(exec_plan_item)}")

        return [create_executable_node(item) for item in self.execution_plan]

    def execute(self, execution_id: str) -> None:
        pass
