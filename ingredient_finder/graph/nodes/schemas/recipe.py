from __future__ import annotations
from pydantic import BaseModel, Field


class RecipeMetadata(BaseModel):
    cuisine: str | None = None
    region_notes: str | None = Field(
        None,
        description="Author's description of regional style, significance or what makes this version distinct.",
    )
    servings: str | None = Field(
        None,
        description="Author verbatim: 'serves 4', 'serves 4-6', 'a large pot', etc.",
    )
    total_time: str | None = Field(
        None,
        description="Author verbatim: 'about an hour', '30 minutes'. None if not mentioned.",
    )


class StepIngredient(BaseModel):
    """An ingredient as it appears in a specific step."""

    name: str = Field(
        description="Author verbatim: preserve regional terms (seeraga samba, kashmiri mirchi, etc.)"
    )
    quantity: str | None = Field(
        None,
        description="Author verbatim: '2 cups', 'a handful', 'enough to coat the pan'",
    )
    prep: str | None = Field(
        None,
        description="Author verbatim: 'finely sliced', 'dry the pieces well'",
    )
    author_note: str | None = Field(
        None,
        description="Author verbatim: commentary on this ingredient in this context. Why it matters, what not to substitute, what to look for.",
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
        description=(
            "Author verbatim: Faithful to author's level of detail, intent and technique"
        ),
    )
    duration: str | None = Field(
        None,
        description=(
            "How the author describes timing/completion for this step, as-is. "
            "Can be time ('about 10 minutes'), sensory ('until oil separates') or both."
        ),
    )
    is_passive: bool = Field(
        False,
        description="True if step involves hands-off waiting: marinating, simmering, soaking, etc",
    )
    ingredients: list[StepIngredient] = Field(
        default_factory=list,
        description="Ingredients introduced or used in this step, with step-specific context",
    )
    sensory_checkpoint: str | None = Field(
        None,
        description="Author verbatim: How the dish should taste/smell/look at the end of this step.",
    )
    author_tips: list[str] = Field(
        default_factory=list,
        description="Author Verbatim: Advice, emphasis, warnings the author gives during this step.",
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

    original: str = Field(description="Ingredient name being substituted")
    substitute: str = Field(description="What the author suggests instead")
    context: str | None = Field(
        None,
        description="Author's commentary: when to use it, tradeoffs, what changes. Verbatim.",
    )


class ExtractedRecipe(BaseModel):
    """
    Pass 1 output: faithful extraction of author-stated information only.

    This is a structured representation of the recipe as the author presents it.
    No inference, no enrichment, no normalization. If the author didn't say it,
    it's not here.
    """

    id: str = Field(description="kebab-case slug, e.g. 'ambur-chicken-biryani'")
    title: str = Field(description="Recipe title as the author names it")
    metadata: RecipeMetadata

    # The recipe IS its steps. Everything else is derived from or secondary to this.
    steps: list[Step]

    # Author-stated substitutions. Usually sparse or empty.
    substitutions: list[AuthorSubstitution] = Field(default_factory=list)

    # Cultural context, history, personal stories, serving traditions.
    # This encodes the "taste identity" of the dish when direct taste
    # descriptions aren't available.
    cultural_context: list[str] = Field(
        default_factory=list,
        description=(
            "Author's commentary on the dish's identity: origin, history, "
            "regional significance, family connection, what makes this version "
            "distinct, how it should be served/eaten. Capture verbatim."
        ),
    )

    # Direct taste/flavor descriptions the author gives about the final dish.
    # "This should be tangy and spicy, not sweet."
    # "The gravy should be thin, not thick like a korma."
    sensory_target: list[str] = Field(
        default_factory=list,
        description=(
            "Author's descriptions of what the finished dish should "
            "taste/feel/look like. The target the cook is aiming for"
        ),
    )


class ExtractedRecipes(BaseModel):
    extracted_recipes: list[ExtractedRecipe]
