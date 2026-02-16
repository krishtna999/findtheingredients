from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage
import operator

from ingredient_finder.graph.nodes.schemas import RecipeDetailsSchema


class VideoMetadata(TypedDict):
    title: str
    description: str
    language: str
    tags: list[str]


class RecipeDetails(TypedDict):
    recipe_raw_text: str
    formatted_recipe_details: RecipeDetailsSchema


class IngredientFinderState(TypedDict):
    video_metadata: VideoMetadata
    video_url: str
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    recipe_details: RecipeDetails
