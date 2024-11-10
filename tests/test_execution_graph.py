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


def test_find_execution_paths_with_edges(basic_graph):
    # Get the execution plan with edges
    edge_plan = basic_graph._find_execution_paths_with_edges()

    def flatten_edge_plan(plan):
        """Helper to flatten the nested structure into a set of edge IDs"""
        edges = set()
        for item in plan:
            if isinstance(item, str):
                edges.add(item)
            elif isinstance(item, list):
                edges.update(flatten_edge_plan(item))
        return edges

    # Get all edges from the plan
    all_edges = flatten_edge_plan(edge_plan)

    # Verify required edges exist
    expected_edges = {
        "__start___to_process_data_1",
        "process_data_to_validate_1",
        "validate_to_escape_1",
        "validate_to_aa_1",
        "escape_to_dd_1",
        "escape_to_cc_1",
        "dd_to_hh_1",
        "cc_to_hh_1",
        "hh_to_prep_1",
        "aa_to_bb_1",
        "bb_to_prep_1",
    }
    assert expected_edges == all_edges

    def verify_structure(plan):
        """Verify the structural properties of the execution plan"""
        if not plan:
            return True

        # If it's a single edge, it should be a string
        if isinstance(plan, str):
            return True

        # If it's a list, verify each element
        if isinstance(plan, list):
            # Parallel paths should be lists within lists
            for item in plan:
                if not (isinstance(item, str) or isinstance(item, list)):
                    return False
                if not verify_structure(item):
                    return False
            return True

        return False

    # Verify the overall structure is valid
    assert verify_structure(edge_plan)

    # Verify some key relationships without caring about exact order
    def find_path_containing(edges, start_edge, end_edge):
        """Check if there exists a path containing both edges in the nested structure"""

        def check_sublist(sublist):
            if isinstance(sublist, str):
                return False

            found_start = False
            found_end = False

            for item in sublist:
                if isinstance(item, str):
                    if item == start_edge:
                        found_start = True
                    if item == end_edge:
                        found_end = True
                elif isinstance(item, list):
                    if check_sublist(item):
                        return True

            return found_start and found_end

        return check_sublist(edges)

    # Verify some key path relationships
    assert find_path_containing(
        edge_plan, "__start___to_process_data_1", "process_data_to_validate_1"
    )

    # Verify parallel paths exist after validate
    parallel_paths_exist = any(
        isinstance(item, list) and len(item) > 1 for item in edge_plan
    )
    assert parallel_paths_exist
