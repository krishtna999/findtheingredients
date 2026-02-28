from __future__ import annotations
from pydantic import BaseModel, Field


class RecipeMetadata(BaseModel):
    cuisine: str | None = None
    region_notes: str | None = Field(
        None,
        description="What makes this version regionally or stylistically distinct, per the author.",
    )
    servings: str | None = Field(
        None,
        description="As the author states it: 'serves 4', 'a large pot', etc.",
    )
    total_time: str | None = Field(
        None,
        description="As the author states it. None if not mentioned.",
    )


class StepIngredient(BaseModel):
    """An ingredient as it appears in a specific step."""

    name: str = Field(
        description="Preserve regional terms (seeraga samba, kashmiri mirchi). Translate to common name only if the original is obscure."
    )
    quantity: str | None = Field(
        None,
        description="As the author states it: '2 cups', 'a handful', 'enough to coat'. None if not specified.",
    )
    prep: str | None = Field(
        None,
        description="How the author says to prepare this ingredient for this step. None if not stated.",
    )
    author_note: str | None = Field(
        None,
        description="Author's commentary on this ingredient in this context — why it matters, what not to substitute, what to look for.",
    )


class CommonMistake(BaseModel):
    """Only mistakes the author explicitly warns about."""

    mistake: str
    consequence: str | None = None
    fix: str | None = None


class Step(BaseModel):
    step: int
    title: str = Field(description="Short descriptive title for the step")
    instruction: str = Field(
        description="What to do in this step. Faithful to the author's intent, technique, and level of detail.",
    )
    duration: str | None = Field(
        None,
        description=(
            "How the author describes timing for this step. "
            "Can be precise ('10 minutes'), approximate ('about 8-10 minutes'), "
            "or open-ended ('until the raw smell goes away'). None if not indicated."
        ),
    )
    is_passive: bool = Field(
        False,
        description="True if step involves hands-off waiting: marinating, simmering, soaking, etc.",
    )
    ingredients: list[StepIngredient] = Field(
        default_factory=list,
        description="Ingredients introduced or used in this step, with step-specific context.",
    )
    sensory_checkpoint: str | None = Field(
        None,
        description="How the dish should look, smell, or taste at the end of this step, per the author. None if not described.",
    )
    author_tips: list[str] = Field(
        default_factory=list,
        description="Advice, emphasis, or warnings the author gives during this step.",
    )
    common_mistakes: list[CommonMistake] = Field(
        default_factory=list,
        description="Only mistakes the author explicitly warns about. Usually empty.",
    )
    equipment: list[str] = Field(
        default_factory=list,
        description="Equipment the author mentions for this step.",
    )


class AuthorSubstitution(BaseModel):
    """A substitution the author explicitly suggests."""

    original: str = Field(description="Ingredient being substituted")
    substitute: str = Field(description="What the author suggests instead")
    context: str | None = Field(
        None,
        description="Author's commentary: when to use it, tradeoffs, what changes.",
    )


class ExtractedRecipe(BaseModel):
    """
    Pass 1 output: faithful extraction of author-stated information only.

    Structured representation of the recipe as the author presents it.
    Rewritten for clarity, but no inference or enrichment.
    If the author didn't say it, it's not here.
    """

    id: str = Field(description="kebab-case slug, e.g. 'ambur-chicken-biryani'")
    title: str = Field(description="Recipe title as the author names it")
    metadata: RecipeMetadata
    steps: list[Step]
    substitutions: list[AuthorSubstitution] = Field(default_factory=list)
    cultural_context: list[str] = Field(
        default_factory=list,
        description=(
            "Author's commentary on the dish's identity: origin, history, "
            "regional significance, family connection, what makes this version "
            "distinct, how it should be served/eaten."
        ),
    )
    sensory_target: list[str] = Field(
        default_factory=list,
        description=(
            "Author's descriptions of what the finished dish should "
            "taste, feel, or look like. The target the cook is aiming for."
        ),
    )


class ExtractedRecipes(BaseModel):
    extracted_recipes: list[ExtractedRecipe]
