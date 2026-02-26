from enum import StrEnum
from langchain_core.messages import HumanMessage, SystemMessage

from ingredient_finder.services.youtube import download_audio
from ingredient_finder.graph.chat_models import model, model_with_tools
from ingredient_finder.graph.tools import transcription_tools_by_name

from ingredient_finder.graph.nodes.schemas.recipe import ExtractedRecipes


class NodeNames(StrEnum):
    TRANSCRIBE_RECIPE_AUDIO = "transcribe_recipe_audio"
    FORMAT_REQUIRED_INGREDIENTS = "format_required_ingredients"
    EXTRACT_RECIPE_FROM_TRANSCRIPT = "extract_recipe_from_transcript"


def transcribe_recipe_audio(state):
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

    return {"recipe_details": {"recipe_raw_text": transcribed_text}}


def format_required_ingredients(state):
    # No-op for now. implement to extract ingredients from the extracted recipes.
    return state


def extract_recipe_from_transcript(state):
    recipe_raw_text = state["recipe_details"]["recipe_raw_text"]
    video_metadata = state["video_metadata"]

    result = model.with_structured_output(ExtractedRecipes).invoke(
        [
            SystemMessage(
                content="""You are a recipe extraction engine. Convert raw recipe text into structured data. Rules:

                1. Extraction only. If the author didn't say it, the field is null or empty. Never infer or enrich with general cooking knowledge.

                2. Preserve the author's voice. Vague quantities ("a good handful"), casual timing ("cook till the raw smell goes"), sensory descriptions, etc - capture as is. 
                   High-confidence translation is fine but do not normalize, convert, or rephrase.

                3. One step = one logical cooking phase as the author presents it. Follow their pacing. Don't split, merge, or reorder steps.

                4. The input may be messy (especially audio transcripts): filler words, repetition, mid-sentence corrections, tangents, sponsor segments. Extract the recipe signal, ignore the noise.
                """
            ),
            HumanMessage(
                content=f"""Extract the recipe from the following raw text.
                <raw_text>
                {recipe_raw_text}
                </raw_text>

                <video_description>
                Watch out for any additional information in the video description provided below:
                {video_metadata.get("description")}
                </video_description>

                <other_metadata>
                title: {video_metadata.get("title")}
                </other_metadata>
                <tags>
                tags: {video_metadata.get("tags")}
                </tags>
                """
            ),
        ]
    )

    return {
        "recipe_details": {
            **state["recipe_details"],
            "recipe_details": result,
        }
    }
