from langgraph.graph import START, StateGraph, END
from ingredient_finder.services.youtube import fetch_metadata
from ingredient_finder.graph.state import IngredientFinderState
from ingredient_finder.graph.edges import should_fetch_recipe_from_audio
from ingredient_finder.graph.nodes import (
    NodeNames,
    fetch_recipe_from_description,
    fetch_recipe_from_audio,
    format_ingredients,
)


agent_builder = StateGraph(IngredientFinderState)

agent_builder.add_node(
    NodeNames.FETCH_RECIPE_FROM_DESCRIPTION, fetch_recipe_from_description
)
agent_builder.add_node(NodeNames.FETCH_RECIPE_FROM_AUDIO, fetch_recipe_from_audio)
agent_builder.add_node(NodeNames.FORMAT_INGREDIENTS, format_ingredients)
agent_builder.add_conditional_edges(
    START,
    should_fetch_recipe_from_audio,
    [
        NodeNames.FETCH_RECIPE_FROM_DESCRIPTION,
        NodeNames.FETCH_RECIPE_FROM_AUDIO,
    ],
)

agent_builder.add_edge(
    NodeNames.FETCH_RECIPE_FROM_DESCRIPTION, NodeNames.FORMAT_INGREDIENTS
)
agent_builder.add_edge(NodeNames.FETCH_RECIPE_FROM_AUDIO, NodeNames.FORMAT_INGREDIENTS)

agent_builder.add_edge(NodeNames.FORMAT_INGREDIENTS, END)

agent = agent_builder.compile()


def preprocess_and_invoke_agent(video_url: str):
    metadata = fetch_metadata(video_url)
    return agent.invoke({"video_metadata": metadata, "video_url": video_url})
