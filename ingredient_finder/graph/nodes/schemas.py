from pydantic import BaseModel, Field


class SingleRecipeDetail(BaseModel):
    """Single recipe with ingredients and key callouts."""

    ingredients: dict[str, str]
    custom_instructions: list[str] = Field(
        description="Non-obvious important tips explicitly mentioned: substitutions, conversions, technique warnings. Never infer."
    )


class RecipeDetailsSchema(BaseModel):
    """Split into separate recipes only if clearly distinct dishes are present."""

    recipes: dict[str, SingleRecipeDetail]
