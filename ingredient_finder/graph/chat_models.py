from langchain.chat_models import init_chat_model
from ingredient_finder.graph.tools import transcription_tools

model = init_chat_model("google_genai:gemini-3-flash-preview", temperature=0)

model_with_tools = model.bind_tools(transcription_tools)
