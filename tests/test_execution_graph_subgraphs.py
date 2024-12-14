from primeGraph.buffer.factory import History, Incremental, LastValue
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState


class SubgraphState(GraphState):
    execution_order: History[str]
    counter: Incremental[int]
    status: LastValue[str]


def test_basic_subgraph_execution():
    state = SubgraphState(execution_order=[], counter=0, status="")
    main_graph = Graph(state=state)

    # Create a subgraph
    @main_graph.subgraph()
    def processing_subgraph():
        subgraph = Graph(state=state)

        @subgraph.node()
        def process_a(state):
            return {"execution_order": "process_a", "counter": 1}

        @subgraph.node()
        def process_b(state):
            return {"execution_order": "process_b", "counter": 2}

        subgraph.add_edge(START, "process_a")
        subgraph.add_edge("process_a", "process_b")
        subgraph.add_edge("process_b", END)

        return subgraph

    # Main graph nodes
    @main_graph.node()
    def start_task(state):
        return {"execution_order": "start_task", "status": "started"}

    @main_graph.node()
    def end_task(state):
        return {"execution_order": "end_task", "status": "completed"}

    # Connect main graph
    main_graph.add_edge(START, "start_task")
    main_graph.add_edge("start_task", "processing_subgraph")
    main_graph.add_edge("processing_subgraph", "end_task")
    main_graph.add_edge("end_task", END)

    main_graph.compile()
    main_graph.start()

    # Verify execution order
    assert state.execution_order == ["start_task", "process_a", "process_b", "end_task"]
    # Verify counter accumulation
    assert state.counter == 3  # 1 + 2
    # Verify final status
    assert state.status == "completed"


def test_parallel_subgraph_execution():
    state = SubgraphState(execution_order=[], counter=0, status="")
    main_graph = Graph(state=state)

    # Create first subgraph
    @main_graph.subgraph()
    def subgraph_a():
        subgraph = Graph(state=state)

        @subgraph.node()
        def process_1(state):
            return {"execution_order": "process_1", "counter": 1}

        @subgraph.node()
        def process_2(state):
            return {"execution_order": "process_2", "counter": 2}

        subgraph.add_edge(START, "process_1")
        subgraph.add_edge("process_1", "process_2")
        subgraph.add_edge("process_2", END)

        return subgraph

    # Create second subgraph
    @main_graph.subgraph()
    def subgraph_b():
        subgraph = Graph(state=state)

        @subgraph.node()
        def process_3(state):
            return {"execution_order": "process_3", "counter": 3}

        @subgraph.node()
        def process_4(state):
            return {"execution_order": "process_4", "counter": 4}

        subgraph.add_edge(START, "process_3")
        subgraph.add_edge("process_3", "process_4")
        subgraph.add_edge("process_4", END)

        return subgraph

    # Main graph nodes
    @main_graph.node()
    def initialize(state):
        return {"execution_order": "initialize", "status": "started"}

    @main_graph.node()
    def finalize(state):
        return {"execution_order": "finalize", "status": "completed"}

    # Connect main graph with parallel subgraphs
    main_graph.add_edge(START, "initialize")
    main_graph.add_edge("initialize", "subgraph_a")
    main_graph.add_edge("initialize", "subgraph_b")
    main_graph.add_edge("subgraph_a", "finalize")
    main_graph.add_edge("subgraph_b", "finalize")
    main_graph.add_edge("finalize", END)

    main_graph.compile()
    main_graph.start()

    # Verify all tasks were executed
    executed_tasks = set(state.execution_order)
    expected_tasks = {
        "initialize",
        "process_1",
        "process_2",
        "process_3",
        "process_4",
        "finalize",
    }
    assert executed_tasks == expected_tasks

    # Verify counter accumulation
    assert state.counter == 10  # 1 + 2 + 3 + 4

    # Verify final status
    assert state.status == "completed"


def test_nested_subgraph_execution():
    state = SubgraphState(execution_order=[], counter=0, status="")
    main_graph = Graph(state=state)

    def nested_subgraph():
        nested = Graph(state=state)

        @nested.node()
        def nested_task(state):
            return {"execution_order": "nested_task", "counter": 1}

        @nested.node()
        def nested_task_2(state):
            return {"execution_order": "nested_task_2", "counter": 2}

        nested.add_edge(START, "nested_task")
        nested.add_edge("nested_task", "nested_task_2")
        nested.add_edge("nested_task_2", END)
        return nested

    # Create parent subgraph containing nested subgraph
    @main_graph.subgraph()
    def parent_subgraph():
        parent = Graph(state=state)

        @parent.node()
        def parent_task(state):
            return {"execution_order": "parent_task", "counter": 2}

        @parent.subgraph()
        def inner_nested():
            return nested_subgraph()

        parent.add_edge(START, "parent_task")
        parent.add_edge("parent_task", "inner_nested")
        parent.add_edge("inner_nested", END)
        return parent

    # Main graph setup
    @main_graph.node()
    def main_task(state):
        return {"execution_order": "main_task", "status": "running"}

    main_graph.add_edge(START, "main_task")
    main_graph.add_edge("main_task", "parent_subgraph")
    main_graph.add_edge("parent_subgraph", END)

    main_graph.compile()
    main_graph.start()

    # Verify execution order
    assert state.execution_order == [
        "main_task",
        "parent_task",
        "nested_task",
        "nested_task_2",
    ]
    # Verify counter accumulation
    assert state.counter == 5  # 1 + 2 + 2
    # Verify status
    assert state.status == "running"
