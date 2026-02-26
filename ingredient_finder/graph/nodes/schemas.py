from pydantic import BaseModel, Field


class SingleRecipeDetail(BaseModel):
    """Single recipe with ingredients and key callouts."""

    ingredients: dict[str, str]


class RecipeDetailsSchema(BaseModel):
    """Split into separate recipes only if clearly distinct dishes are present."""

    recipes: dict[str, SingleRecipeDetail]
