import pytest

from tiny_graph.constants import END, START
from tiny_graph.graph.executable import ExecutableNode, Graph


@pytest.fixture
def basic_graph():
    simple_graph = Graph()

    # Define some example actions
    @simple_graph.node()
    def escape():
        print("Starting workflow")

    @simple_graph.node()
    def process_data():
        print("Processing data")

    @simple_graph.node()
    def validate():
        print("Validating results")

    @simple_graph.node()
    def aa():
        print("Validating results")

    @simple_graph.node()
    def bb():
        print("Validating results")

    @simple_graph.node()
    def dd():
        print("Validating results")

    @simple_graph.node()
    def cc():
        print("Validating results")

    @simple_graph.node()
    def hh():
        print("Validating results")

    @simple_graph.node()
    def prep():
        print("Workflow complete")

    # Add edges to create workflow
    simple_graph.add_edge(START, "process_data")
    simple_graph.add_edge("process_data", "validate")
    simple_graph.add_edge("validate", "escape")
    simple_graph.add_edge("escape", "dd")
    simple_graph.add_edge("escape", "cc")
    simple_graph.add_edge("cc", "hh")
    simple_graph.add_edge("dd", "hh")
    simple_graph.add_edge("hh", "prep")
    simple_graph.add_edge("validate", "aa")
    simple_graph.add_edge("aa", "bb")
    simple_graph.add_edge("bb", "prep")
    simple_graph.add_edge("prep", END)

    simple_graph.compile()

    return simple_graph


def extract_executable_nodes_info(executable_node):
    if len(executable_node.task_list) <= 1:
        return (executable_node.task_list[0], executable_node.execution_type)
    else:
        return [
            extract_executable_nodes_info(task) for task in executable_node.task_list
        ]


def test_execution_plan_conversion(basic_graph):
    # Test sequential execution
    basic_graph.execution_plan = ["process_data", "validate"]
    result = basic_graph._convert_execution_plan()

    assert len(result) == 2
    assert all(isinstance(node, ExecutableNode) for node in result)
    assert result[0].node_name == "process_data"
    assert result[0].execution_type == "sequential"
    assert len(result[0].task_list) == 1

    # Test parallel execution
    basic_graph.execution_plan = [["escape", "aa"]]
    result = basic_graph._convert_execution_plan()

    assert len(result) == 1
    assert result[0].node_name == "group_escape"
    assert result[0].execution_type == "parallel"
    assert len(result[0].task_list) == 2

    # Test mixed execution
    basic_graph.execution_plan = ["process_data", ["escape", "cc"], "prep"]
    result = basic_graph._convert_execution_plan()

    assert len(result) == 3
    assert result[0].execution_type == "sequential"
    assert result[1].execution_type == "parallel"
    assert result[2].execution_type == "sequential"
    assert len(result[1].task_list) == 2


def test_execution_plan_invalid_input(basic_graph):
    # Test invalid input
    basic_graph.execution_plan = [None]
    with pytest.raises(ValueError):
        basic_graph._convert_execution_plan()


# # Test basic graph creation and properties
# def test_execution_order(basic_graph):
#     pass


# def test_state_updates(basic_graph):
#     pass


# def test_concurrency(basic_graph):
#     pass


# def test_checkpoints(basic_graph):
#     pass


# def test_jump_execution(basic_graph):
#     pass


# def test_interrupt_execution(basic_graph):
#     pass


# def test_error_handling(basic_graph):
#     pass
