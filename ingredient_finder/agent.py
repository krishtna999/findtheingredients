from langgraph.graph import START, StateGraph, END
from ingredient_finder.services.youtube import fetch_metadata
from ingredient_finder.graph.state import RecipeExtractorState
from ingredient_finder.graph.nodes import (
    NodeNames,
    transcribe_recipe_audio,
    extract_recipe_from_transcript,
)


agent_builder = StateGraph(RecipeExtractorState)

agent_builder.add_node(NodeNames.TRANSCRIBE_RECIPE_AUDIO, transcribe_recipe_audio)
agent_builder.add_node(
    NodeNames.EXTRACT_RECIPE_FROM_TRANSCRIPT, extract_recipe_from_transcript
)

agent_builder.add_edge(START, NodeNames.TRANSCRIBE_RECIPE_AUDIO)
agent_builder.add_edge(
    NodeNames.TRANSCRIBE_RECIPE_AUDIO, NodeNames.EXTRACT_RECIPE_FROM_TRANSCRIPT
)

agent_builder.add_edge(NodeNames.EXTRACT_RECIPE_FROM_TRANSCRIPT, END)

agent = agent_builder.compile()


# ADR: Should be the first NODE in the graph. That way we can add a validation check as well.
# This NODE will preprocess and validate if the video has a valid recipe or not (based on metadata, etc).
# It makes no functional difference to do this outside the graph vs doing inside the graph.
# But I guess the throwing it in the graph will also provide visibility via automated Langsmith analytics.
# So I guess club all-things agent related under Langgraph to utilize the surrounding ecosystem.
def preprocess_and_invoke_agent(video_url: str):
    metadata = fetch_metadata(video_url)
    return agent.invoke({"video_metadata": metadata, "video_url": video_url})
