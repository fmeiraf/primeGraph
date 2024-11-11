import concurrent.futures
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

    def execute(
        self, execution_id: str = None, timeout: Union[int, float] = 60 * 5
    ) -> None:
        """Execute the graph with concurrent and sequential execution based on the execution plan.

        Args:
            execution_id: Unique identifier for this execution
        """

        def execute_node(node: ExecutableNode) -> None:
            """Execute a single node or group of nodes with proper concurrency handling."""
            if node.execution_type == "sequential":
                # For sequential nodes, execute each task with timeout
                for task in node.task_list:
                    try:
                        # Use ThreadPoolExecutor for timeout even in sequential case
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(task)
                            # Wait for the task to complete with timeout
                            future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(
                            f"Execution timeout in sequential node {node.node_name}"
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Error in node {node.node_name}: {str(e)}"
                        ) from e
            else:  # parallel execution
                # Create a thread pool for parallel execution
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit all tasks/nested nodes
                    futures = []
                    for task in node.task_list:
                        if isinstance(task, ExecutableNode):
                            # Handle nested nodes recursively
                            futures.append(executor.submit(execute_node, task))
                        else:
                            futures.append(executor.submit(task))

                    # Wait for all tasks to complete with timeout
                    try:
                        concurrent.futures.wait(
                            futures,
                            timeout=timeout,
                            return_when=concurrent.futures.ALL_COMPLETED,
                        )

                        # Check for exceptions
                        for future in futures:
                            if future.exception():
                                raise RuntimeError(
                                    f"Error in parallel execution of {node.node_name}: {str(future.exception())}"
                                ) from future.exception()

                    except concurrent.futures.TimeoutError:
                        # Cancel all pending tasks if timeout occurs
                        for future in futures:
                            future.cancel()
                        raise TimeoutError(
                            f"Execution timeout in node {node.node_name}"
                        )

        # Convert the execution plan and execute
        execution_plan = self._convert_execution_plan()
        for node in execution_plan:
            execute_node(node)
