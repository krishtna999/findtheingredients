from enum import StrEnum
from langchain_core.messages import HumanMessage

from ingredient_finder.services.youtube import download_audio
from ingredient_finder.graph.chat_models import model, model_with_tools
from ingredient_finder.graph.tools import transcription_tools_by_name

from ingredient_finder.graph.nodes.schemas import RecipeDetailsSchema


class NodeNames(StrEnum):
    FETCH_RECIPE_FROM_DESCRIPTION = "fetch_recipe_from_description"
    FETCH_RECIPE_FROM_AUDIO = "fetch_recipe_from_audio"
    FORMAT_INGREDIENTS = "format_ingredients"


def fetch_recipe_from_description(state):
    description = state["video_metadata"]["description"]

    result = model.invoke(
        [
            HumanMessage(
                content=(
                    "Trim the video description to only include the ingredients and instructions."
                    "Remove any extra information.\n\n"
                    f"{description}"
                )
            ),
        ]
    )

    return {
        "recipe_details": {**state["recipe_details"], "recipe_raw_text": result.content}
    }


def fetch_recipe_from_audio(state):
    audio_path = download_audio(state["video_url"], state["video_metadata"]["title"])
    tags = state["video_metadata"]["tags"]
    language = state["video_metadata"]["language"]

    # Ideally, the below decision should be backed up with another tool that figures out the language from the audio to make it robust.
    # But I guess this is fine for now and we deal with that problem if the tags are not enough.
    # Also this could be really done without the model reasoning about tools, but hey im just learning langgraph so WHY NOT ?
    selected_tools = model_with_tools.invoke(
        [
            HumanMessage(
                content=(
                    "Carefully reason and choose the right language-tool to transcribe (and translate) the audio based on the following metadata: \n"
                    f"tags: {tags}\n"
                    f"language: {language}\n"
                    f"saved_audio_path: {audio_path}"
                )
            ),
        ]
    )

    if len(selected_tools.tool_calls) == 0:
        raise Exception(f"No suitable tool found for the given tags: {tags}")

    tool_call = selected_tools.tool_calls[0]
    tool = transcription_tools_by_name[tool_call["name"]]
    transcribed_text = tool.invoke(tool_call["args"])

    trimmed_recipe = model.invoke(
        [
            HumanMessage(
                content=(
                    "Trim the transcribed text to only include the cooking instructions, ingredients and important tips/callouts.."
                    "Remove any extra information.\n\n"
                    f"Recipe: {transcribed_text}"
                )
            ),
        ]
    )

    return {"recipe_details": {"recipe_raw_text": trimmed_recipe.content}}


def format_ingredients(state):
    recipe_raw_text = state["recipe_details"]["recipe_raw_text"]
    result = model.with_structured_output(RecipeDetailsSchema).invoke(
        [
            HumanMessage(
                content=(
                    "Extract the ingredients and important tips from the following recipe raw text."
                    "Return each ingredient with its quantity.\n\n"
                    f"{recipe_raw_text}"
                )
            ),
        ]
    )

    return {
        "recipe_details": {
            **state["recipe_details"],
            "formatted_recipe_details": result.recipes,
        }
    }
