from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage
import operator

from ingredient_finder.graph.nodes.schemas.recipe import ExtractedRecipe


class VideoMetadata(TypedDict):
    title: str
    description: str
    language: str
    tags: list[str]


class RecipeDetails(TypedDict):
    recipe_raw_text: str
    extracted_recipe: ExtractedRecipe
    required_ingredients: list[str]


class RecipeExtractorState(TypedDict):
    video_metadata: VideoMetadata
    recipe_details: RecipeDetails
    video_url: str
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
