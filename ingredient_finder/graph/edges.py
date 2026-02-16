from typing import Literal

from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from ingredient_finder.graph.state import IngredientFinderState
from ingredient_finder.graph.chat_models import model
from ingredient_finder.graph.nodes import NodeNames


class HasIngredients(BaseModel):
    """Whether a video description contains cooking ingredients."""

    has_ingredients: bool


def should_fetch_recipe_from_audio(
    state: IngredientFinderState,
) -> Literal[
    NodeNames.FETCH_RECIPE_FROM_AUDIO, NodeNames.FETCH_RECIPE_FROM_DESCRIPTION
]:
    description = state["video_metadata"]["description"]

    result = model.with_structured_output(HasIngredients).invoke(
        [
            HumanMessage(
                content=(
                    "Does this video description contain valid cooking ingredients?"
                    "Ignore links, timestamps, and general commentary.\n\n"
                    f"Video description: {description}"
                )
            ),
        ]
    )

    if result.has_ingredients:
        return NodeNames.FETCH_RECIPE_FROM_DESCRIPTION
    else:
        return NodeNames.FETCH_RECIPE_FROM_AUDIO
