from dotenv import load_dotenv

load_dotenv()

import json
import os
from pydantic import BaseModel
from ingredient_finder.agent import preprocess_and_invoke_agent

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pydantic_serializer(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    # Starting with pongal can never go wrong !
    # input = "https://www.youtube.com/watch?v=WR5JJP5MyN4"
    input = "https://www.youtube.com/watch?v=3Hn6iwieyX4"
    agent_result = preprocess_and_invoke_agent(input)

    with open(f"{OUTPUT_DIR}/{agent_result['video_metadata']['title']}.json", "w") as f:
        json.dump(agent_result, f, indent=2, default=pydantic_serializer)
