from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from primeGraph.buffer.factory import History, LastValue
from primeGraph.checkpoint.base import CheckpointData
from primeGraph.checkpoint.postgresql import PostgreSQLStorage
from primeGraph.constants import END, START
from primeGraph.graph.executable import Graph
from primeGraph.models.state import GraphState

# Requires you to be running the docker from primeGraph/docker


@pytest.fixture
def postgres_storage():
  storage = PostgreSQLStorage.from_config(
    host="localhost",
    port=5432,
    user="primegraph",
    password="primegraph",
    database="primegraph",
  )
  assert storage.check_schema(), "Schema is not valid"
  return storage


class StateForTest(GraphState):
  value: LastValue[int]
  text: LastValue[Optional[str]] = None


def test_save_and_load_checkpoint(postgres_storage):
  # Initialize
  state = StateForTest(value=42, text="initial")
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  # Save checkpoint
  checkpoint_data = CheckpointData(chain_id=graph.chain_id, chain_status=graph.chain_status)
  checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  # Load checkpoint
  loaded_state = graph.checkpoint_storage.load_checkpoint(state, graph.chain_id, checkpoint_id)

  serialized_data = state.__class__.model_validate_json(loaded_state.data)

  assert serialized_data.value == state.value
  assert serialized_data.text == state.text


def test_list_checkpoints(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  # Save multiple checkpoints
  checkpoint_data = CheckpointData(chain_id=graph.chain_id, chain_status=graph.chain_status)
  checkpoint_1 = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  state.value = 43
  checkpoint_2 = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)

  checkpoints = graph.checkpoint_storage.list_checkpoints(graph.chain_id)
  assert len(checkpoints) == 2
  assert checkpoint_1 in [c.checkpoint_id for c in checkpoints]
  assert checkpoint_2 in [c.checkpoint_id for c in checkpoints]


def test_delete_checkpoint(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  # Save and then delete a checkpoint
  checkpoint_data = CheckpointData(chain_id=graph.chain_id, chain_status=graph.chain_status)
  checkpoint_id = graph.checkpoint_storage.save_checkpoint(state, checkpoint_data)
  assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 1

  graph.checkpoint_storage.delete_checkpoint(graph.chain_id, checkpoint_id)
  assert len(graph.checkpoint_storage.list_checkpoints(graph.chain_id)) == 0


class StateForTestWithHistory(GraphState):
  execution_order: History[str]


@pytest.mark.asyncio
async def test_resume_with_checkpoint_load(postgres_storage):
  state = StateForTestWithHistory(execution_order=[])
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  @graph.node()
  def task1(state):
    return {"execution_order": "task1"}

  @graph.node()
  def task2(state):
    return {"execution_order": "task2"}

  @graph.node()
  def task3(state):
    return {"execution_order": "task3"}

  @graph.node(interrupt="before")
  def task4(state):
    return {"execution_order": "task4"}

  graph.add_edge(START, "task1")
  graph.add_edge("task1", "task2")
  graph.add_edge("task2", "task3")
  graph.add_edge("task3", "task4")
  graph.add_edge("task4", END)
  graph.compile()

  
  # Start new chain to test load
  chain_id = await graph.execute()

  # Load first chain state
  graph.load_from_checkpoint(chain_id)

  # Resume execution
  await graph.resume()
  assert all(task in graph.state.execution_order for task in ["task1", "task2", "task3", "task4"])


def test_nonexistent_checkpoint(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  with pytest.raises(KeyError):
    graph.checkpoint_storage.load_checkpoint(state, graph.chain_id, "nonexistent")


def test_nonexistent_chain(postgres_storage):
  state = StateForTest(value=42)
  graph = Graph(state=state, checkpoint_storage=postgres_storage)

  with pytest.raises(KeyError):
    graph.checkpoint_storage.load_checkpoint(state, "nonexistent", "some_checkpoint")


def test_load_checkpoint_with_nested_basemodels(postgres_storage):
    class BreakdownInstruction(BaseModel):
      step_name: str = Field(description="The name of the step")
      step_instructions: str = Field(description="The instructions for the step")

    class StateWithInstructions(GraphState):
        instructions: LastValue[List[BreakdownInstruction]] = Field(default_factory=list)

    # Create initial state with BreakdownInstruction instances
    instructions = [
        BreakdownInstruction(
            step_name="Research SpaceX Missions",
            step_instructions="Start by researching SpaceX's upcoming missions..."
        ),
        BreakdownInstruction(
            step_name="Budget Planning",
            step_instructions="Calculate the estimated costs..."
        )
    ]
    
    # First graph instance - create and save state
    initial_state = StateWithInstructions(instructions=instructions)
    first_graph = Graph(state=initial_state, checkpoint_storage=postgres_storage)
    
    # Save initial state with checkpoint
    checkpoint_data = CheckpointData(
        chain_id=first_graph.chain_id,
        chain_status=first_graph.chain_status
    )
    first_graph.checkpoint_storage.save_checkpoint(first_graph.state, checkpoint_data)
    chain_id = first_graph.chain_id

    # Create a fresh graph instance with empty state
    fresh_state = StateWithInstructions(instructions=[])
    fresh_graph = Graph(state=fresh_state, checkpoint_storage=postgres_storage)
    
    # Load the checkpoint into the fresh graph
    fresh_graph.load_from_checkpoint(chain_id)

    # Verify the loaded state matches the original
    assert len(fresh_graph.state.instructions) == len(instructions)
    
    for original_instruction, loaded_instruction in zip(instructions, fresh_graph.state.instructions):
        assert isinstance(loaded_instruction, BreakdownInstruction)
        assert loaded_instruction.step_name == original_instruction.step_name
        assert loaded_instruction.step_instructions == original_instruction.step_instructions

    # Verify chain status and ID were properly restored
    assert fresh_graph.chain_id == chain_id
    assert fresh_graph.chain_status == first_graph.chain_status


def test_page_insertion_serialization(postgres_storage):
    import json
    from enum import Enum
    from typing import Any, List, Optional

    from pydantic import BaseModel

    class ColorWithBackground(Enum):
        DEFAULT = "default"

    class RichTextAnnotation(BaseModel):
        bold: bool
        italic: bool
        strikethrough: bool
        underline: bool
        code: bool
        color: ColorWithBackground

    class TextContent(BaseModel):
        content: str
        link: Optional[str] = None

    class RichText(BaseModel):
        type: str
        annotations: RichTextAnnotation
        plain_text: str
        href: Optional[str] = None
        text: TextContent
        equation: Optional[Any] = None
        mention: Optional[Any] = None

    class ParagraphBlockData(BaseModel):
        rich_text: List[RichText]
        color: Optional[Any] = None
        children: Optional[Any] = None

    class Block(BaseModel):
        object: str
        type: str
        paragraph: Optional[ParagraphBlockData] = None
        divider: Optional[dict] = None

    class DatabaseParent(BaseModel):
        type: str
        database_id: str

    class Option(BaseModel):
        name: str
        color: Optional[Any] = None
        description: Optional[Any] = None

    class StatusProperty(BaseModel):
        description: Optional[Any] = None
        type: Optional[Any] = None
        status: Option

    class SelectProperty(BaseModel):
        description: Optional[Any] = None
        type: Optional[Any] = None
        select: Option

    class MultiSelectProperty(BaseModel):
        description: Optional[Any] = None
        type: Optional[Any] = None
        multi_select: List[Option]

    class TitleProperty(BaseModel):
        description: Optional[Any] = None
        type: Optional[Any] = None
        title: List[RichText]

    class DatabaseProperties(BaseModel):
        Number_test: Optional[Any] = None
        Date_test: Optional[Any] = None
        Formula_test: Optional[Any] = None
        Email_test: Optional[Any] = None
        Text_test: Optional[Any] = None
        Person_test: Optional[Any] = None
        Phone_test: Optional[Any] = None
        Status_test: Optional[StatusProperty] = None
        Select_test: Optional[SelectProperty] = None
        Url_test: Optional[Any] = None
        Files_test: Optional[Any] = None
        Checkbox_test: Optional[Any] = None
        Multi_select_test: Optional[MultiSelectProperty] = None
        Name: Optional[TitleProperty] = None

    class PageInsertion(BaseModel):
        object: str
        icon: Optional[Any] = None
        parent: DatabaseParent
        children: List[Block]
        archived: Optional[Any] = None
        in_trash: Optional[Any] = None
        url: Optional[Any] = None
        public_url: Optional[Any] = None
        properties: DatabaseProperties

    # Construct the example object as provided
    page_insertion = PageInsertion(
        object="page",
        icon=None,
        parent=DatabaseParent(type="database_id", database_id="your_database_id"),
        children=[
            Block(
                object="block",
                type="paragraph",
                paragraph=ParagraphBlockData(
                    rich_text=[
                        RichText(
                            type="text",
                            annotations=RichTextAnnotation(
                                bold=False,
                                italic=False,
                                strikethrough=False,
                                underline=False,
                                code=False,
                                color=ColorWithBackground.DEFAULT
                            ),
                            plain_text="**Research Opportunities:** Investigate potential pathways to participate in moon missions, either through established government space agencies like NASA or private enterprises such as SpaceX.",
                            href=None,
                            text=TextContent(
                                content="**Research Opportunities:** Investigate potential pathways to participate in moon missions, either through established government space agencies like NASA or private enterprises such as SpaceX.",
                                link=None
                            ),
                            equation=None,
                            mention=None
                        )
                    ],
                    color=None,
                    children=None
                )
            ),
            Block(
                object="block",
                type="divider",
                divider={}
            ),
            Block(
                object="block",
                type="paragraph",
                paragraph=ParagraphBlockData(
                    rich_text=[
                        RichText(
                            type="text",
                            annotations=RichTextAnnotation(
                                bold=False,
                                italic=False,
                                strikethrough=False,
                                underline=False,
                                code=False,
                                color=ColorWithBackground.DEFAULT
                            ),
                            plain_text="**Application Preparation:** Develop a strong application that emphasizes your educational qualifications, physical fitness, and your drive to be part of a moon mission.",
                            href=None,
                            text=TextContent(
                                content="**Application Preparation:** Develop a strong application that emphasizes your educational qualifications, physical fitness, and your drive to be part of a moon mission.",
                                link=None
                            ),
                            equation=None,
                            mention=None
                        )
                    ],
                    color=None,
                    children=None
                )
            ),
            Block(
                object="block",
                type="divider",
                divider={}
            ),
            Block(
                object="block",
                type="paragraph",
                paragraph=ParagraphBlockData(
                    rich_text=[
                        RichText(
                            type="text",
                            annotations=RichTextAnnotation(
                                bold=False,
                                italic=False,
                                strikethrough=False,
                                underline=False,
                                code=False,
                                color=ColorWithBackground.DEFAULT
                            ),
                            plain_text="**Networking:** Connect with professionals in the space field to gather advice and enhance your selection prospects.",
                            href=None,
                            text=TextContent(
                                content="**Networking:** Connect with professionals in the space field to gather advice and enhance your selection prospects.",
                                link=None
                            ),
                            equation=None,
                            mention=None
                        )
                    ],
                    color=None,
                    children=None
                )
            )
        ],
        archived=None,
        in_trash=None,
        url=None,
        public_url=None,
        properties=DatabaseProperties(
            Number_test=None,
            Date_test=None,
            Formula_test=None,
            Email_test=None,
            Text_test=None,
            Person_test=None,
            Phone_test=None,
            Status_test=StatusProperty(
                description=None,
                type=None,
                status=Option(name="Not started", color=None, description=None)
            ),
            Select_test=SelectProperty(
                description=None,
                type=None,
                select=Option(name="Engineering", color=None, description=None)
            ),
            Url_test=None,
            Files_test=None,
            Checkbox_test=None,
            Multi_select_test=MultiSelectProperty(
                description=None,
                type=None,
                multi_select=[Option(name="one", color=None, description=None), Option(name="two", color=None, description=None)]
            ),
            Name=TitleProperty(
                description=None,
                type=None,
                title=[
                    RichText(
                        type="text",
                        annotations=RichTextAnnotation(
                            bold=False,
                            italic=False,
                            strikethrough=False,
                            underline=False,
                            code=False,
                            color=ColorWithBackground.DEFAULT
                        ),
                        plain_text="Moon Mission Participation Plan",
                        href=None,
                        text=TextContent(
                            content="Moon Mission Participation Plan",
                            link=None
                        ),
                        equation=None,
                        mention=None
                    )
                ]
            )
        )
    )

    # Convert the object using the storage conversion method and attempt JSON serialization
    converted = postgres_storage._convert_sets_to_lists(page_insertion)
    serialized_json = json.dumps(converted)

    # Assert that the enum value has been converted to its value 'default'
    assert "default" in serialized_json
