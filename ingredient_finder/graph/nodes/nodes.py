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
                    "Trim the video description to only include the ingredients, instructions and callouts."
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

    # ADR: Ideally, the below decision should be backed up with another tool that figures out the language from the audio to make it robust.
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

    # ADR: Don't feel too good about relying on a tool_calls truthy-check to determine if a tool was selected. Kinda feels hacky (maybe cuz I come from Java lmao).
    # I would prefer a flag or something more concrete but then I guess this is how things are done in python and it's just fine.
    if not selected_tools.tool_calls:
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


# ADR: This can maybe be done in the same prompt/LLM call in the previous fetch_recipe_* nodes which will lead to 1 less LLM call overall.
# But imo, it's better to keep a separate node for this (unless it breaks the cost all-together).
# Since it follows DRY it is easily re-used, unit-tested, etc etc. Also leads to better analytics, tracking and debugging via Langgraph/smith.
# Not to mention the fact that you can have a less expensive model for this task (which is just formatting text to JSON).
#
# I guess finding the right level of SRP is key to balancing maintenance/engineering effort and AI costs.
# If you focus on SRP too much, you end up with extra LLM calls.
# If you don't focus on it enough, you end with a monolithic, hard to track/debug/maintain codebase.
# But I'd still lean towards more SRP (esp. with more funding xD).


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
