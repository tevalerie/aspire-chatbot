"""The shape of a lesson, and the loader that refuses to start without one."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.safety import vocab

logger = logging.getLogger(__name__)

Band = Literal["5-8", "9-12", "13-15", "16-18", "adult"]

#: Bands youngest first.
BANDS: tuple[Band, ...] = ("5-8", "9-12", "13-15", "16-18", "adult")

CONTENT_DIR = Path(__file__).resolve().parent / "content"


def band_index(band: str) -> int:
    try:
        return BANDS.index(band)  # type: ignore[arg-type]
    except ValueError:
        return -1


def for_band(mapping: dict[str, Any], band: str) -> Any | None:
    """The entry for `band`, falling back to the nearest YOUNGER band."""
    index = band_index(band)
    if index < 0:
        return None
    for step in range(index, -1, -1):
        if BANDS[step] in mapping:
            return mapping[BANDS[step]]
    return None


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Concept(_Node):
    """One idea, with the bands that may meet it and the words it introduces."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(max_length=80)
    band_min: Band
    band_max: Band = "adult"
    prerequisites: list[str] = Field(default_factory=list)
    #: The words this concept teaches.
    vocabulary: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bands_are_ordered(self) -> "Concept":
        if band_index(self.band_min) > band_index(self.band_max):
            raise ValueError(f"{self.id}: band_min is above band_max")
        return self

    @model_validator(mode="after")
    def _vocabulary_is_permitted_at_band_min(self) -> "Concept":
        """A concept may not teach a word its youngest band cannot hear.

        TEACHING SCOPE, and it is the reason a lesson can exist at all.

        The client's ruling: for educational purposes the ban lifts. It had not
        reached here, and here fails HARDER than anywhere else -- this is a
        load-time validator, so a concept whose vocabulary contained
        `investment` at 5-8, or `compound` or `dividend` at 9-12, did not get
        stripped at reply time. **The module refused to load**, and the tutor
        had nothing to teach at all.

        Which put the ladder in an impossible position: `investment` is banned
        at 5-8 precisely so a guide does not wander into it, and a LESSON about
        it is the one context where the word is the point. A curriculum that
        cannot name what it teaches is not a safer curriculum.

        So a concept is read under TEACHING scope -- the wider of the two, and
        the right one here, because a concept is a lesson. It lifts
        `vocab.TEACHING_TERMS`, which is the programme's own vocabulary plus
        `loan`: ASPIRE lends nobody anything, so `loan` is not a programme term,
        and a lesson about borrowing cannot be written without it.

        `_GENERAL_BAN` still rejects a module at load, and so does anything
        outside that set -- a concept about `inflation` at 5-8 is still refused,
        because that is neither what the programme calls itself nor what a
        lesson at that band is for.
        """
        for word in self.vocabulary:
            if vocab.check(word, self.band_min, teaching_scope=True):
                raise ValueError(
                    f"{self.id}: vocabulary {word!r} is banned at {self.band_min}"
                )
        return self


class CheckQuestion(_Node):
    """One check-for-understanding question."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{1,63}$")
    prompt: dict[str, str]
    options: list[str] = Field(default_factory=list, max_length=4)
    #: Index into `options`. None for a free-text question.
    answer: int | None = None
    #: Concept words that count as a correct free-text answer, lowercase.
    accept: list[str] = Field(default_factory=list)
    graded: bool = True
    #: The three rungs, band-keyed.
    hints: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _an_answer_indexes_an_option(self) -> "CheckQuestion":
        if self.answer is not None and not 0 <= self.answer < len(self.options):
            raise ValueError(f"{self.id}: answer does not index an option")
        if self.graded and self.answer is None and not self.accept:
            raise ValueError(
                f"{self.id}: a graded question needs either an answer index or "
                "an accept list"
            )
        return self


class Misconception(_Node):
    """What learners get wrong, and the sentence to say when they do."""

    wrong: str = Field(max_length=200)
    say: str = Field(max_length=200)


class EducatorGuide(_Node):
    """One lesson, rendered for somebody who will teach it."""

    timing_minutes: int = Field(ge=5, le=120)
    materials: list[str] = Field(default_factory=list)
    misconceptions: list[Misconception] = Field(default_factory=list)
    extension: str = Field(default="", max_length=240)


class GuardianGuide(_Node):
    """One lesson, rendered for somebody who will do it alongside a child."""

    doing: str = Field(max_length=200)
    do_together: str = Field(max_length=300)
    ask_them: str = Field(max_length=200)
    got_it_sounds_like: str = Field(max_length=200)


class Guide(_Node):
    """The adult renderings of a lesson.

    BOTH MAPS ARE KEYED BY THE BAND OF THE CHILD, not of the reader. A guardian
    and an educator both read at `adult`; what changes is the age of the learner
    they are asking about. Pass the child's band to `for_band`, never the
    reader's, or every lookup resolves to nothing.
    """

    educator: dict[str, EducatorGuide] = Field(default_factory=dict)
    guardian: dict[str, GuardianGuide] = Field(default_factory=dict)


#: The six persona keys a lesson may carry a voice for.
#:
#: Duplicated from `app.graph.access.PERSONAS` rather than imported, because the
#: curriculum must load without pulling the graph in behind it. A test asserts
#: the two stay equal, so the duplication cannot drift silently.
#:
#: KEYS, never labels. "Stella" was renamed to "Skye" in the UI while the key
#: stayed `stella`; a lesson keyed on a display name would have been orphaned by
#: that rename with nothing to say so.
PERSONA_KEYS: tuple[str, ...] = ("stella", "kaleb", "orion", "aurora", "nova", "guest")


class PersonaVoice(_Node):
    """How one persona says a lesson, over and above what its band requires.

    A band is a reading level. A persona is a voice. Skye and Kaleb differ by
    band AND voice; Imani, Azuri and Guest are all `adult` and differ only by
    voice, which is exactly why the band map could not express them and every
    adult fell through to the 13-15 content.

    `band` is declared so the authored words can be checked against the
    vocabulary ladder at load time. It does NOT set the reader's band -- the
    session does that, and the caps and gates read it from there.
    """

    band: Band | None = None
    role: str | None = Field(default=None, max_length=32)
    teach_points: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    joke: str | None = Field(default=None, max_length=200)
    tone: str | None = Field(default=None, max_length=80)


class Lesson(_Node):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    module_id: str
    concept_id: str
    order: int = Field(ge=1)
    objective: str = Field(max_length=200)
    #: Band-keyed. Each entry is the ordered sequence of things to say.
    teach_points: dict[str, list[str]]
    #: Band-keyed.
    examples: dict[str, list[str]]
    check_questions: list[CheckQuestion] = Field(min_length=1)
    suggested_widget_kind: str | None = None
    #: Band-keyed by the CHILD's band. See `Guide`.
    guide: Guide = Field(default_factory=Guide)
    #: Persona-keyed, and consulted BEFORE the band map. See `PersonaVoice`.
    persona_voice: dict[str, PersonaVoice] = Field(default_factory=dict)

    @field_validator("persona_voice")
    @classmethod
    def _persona_keys_are_real(cls, value: dict[str, PersonaVoice]) -> dict[str, PersonaVoice]:
        unknown = set(value) - set(PERSONA_KEYS)
        if unknown:
            raise ValueError(
                f"persona_voice: unknown persona key(s) {sorted(unknown)}. "
                f"Use a key, not a label: {list(PERSONA_KEYS)}"
            )
        return value

    @field_validator("guide")
    @classmethod
    def _guide_bands_are_real(cls, value: "Guide") -> "Guide":
        for name, mapping in (("educator", value.educator), ("guardian", value.guardian)):
            unknown = set(mapping) - set(BANDS)
            if unknown:
                raise ValueError(f"guide.{name}: unknown age band(s): {sorted(unknown)}")
        return value

    def educator_guide(self, child_band: str) -> "EducatorGuide | None":
        """The teaching notes for a class at this band."""
        return for_band(self.educator_map(), child_band)

    def guardian_guide(self, child_band: str) -> "GuardianGuide | None":
        """What to say to the adult of a child at this band."""
        return for_band(self.guide.guardian, child_band)

    def educator_map(self) -> dict[str, "EducatorGuide"]:
        return self.guide.educator

    @field_validator("teach_points", "examples")
    @classmethod
    def _bands_are_real(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        unknown = set(value) - set(BANDS)
        if unknown:
            raise ValueError(f"unknown age band(s): {sorted(unknown)}")
        return value

    def teach_for(self, band: str, persona: str | None = None) -> list[str]:
        """This persona's words if it has any, else the band's.

        One shape either way -- a list of strings. The persona layer must not
        make callers handle two return types.
        """
        voice = self.persona_voice.get(persona or "")
        if voice is not None and voice.teach_points:
            # A voice is authored AT a band (orion's at 13-15). A reader ABOVE
            # that band -- Zion's second rung, 16-18 -- outgrows the voice text
            # and takes the band map instead, where the older wording lives.
            # Without this, a seventeen-year-old was taught in the 13-15 voice.
            outgrown = (
                voice.band is not None
                and band_index(band) > band_index(voice.band)
                and for_band(self.teach_points, band) is not None
            )
            if not outgrown:
                return list(voice.teach_points)
        return list(for_band(self.teach_points, band) or [])

    def examples_for(self, band: str, persona: str | None = None) -> list[str]:
        voice = self.persona_voice.get(persona or "")
        if voice is not None and voice.examples:
            # Same rung rule as `teach_for`: a reader above the voice's band
            # takes the band map's older examples.
            outgrown = (
                voice.band is not None
                and band_index(band) > band_index(voice.band)
                and for_band(self.examples, band) is not None
            )
            if not outgrown:
                return list(voice.examples)
        return list(for_band(self.examples, band) or [])

    def joke_for(self, persona: str | None) -> str | None:
        """The aside this persona would make, or None. Never falls back."""
        voice = self.persona_voice.get(persona or "")
        return voice.joke if voice is not None else None


class Module(_Node):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    title: str = Field(max_length=120)
    order: int = Field(ge=1)
    band_min: Band
    band_max: Band = "adult"
    concepts: list[Concept] = Field(default_factory=list)
    lessons: list[Lesson] = Field(min_length=1)

    @model_validator(mode="after")
    def _lessons_are_consistent(self) -> "Module":
        orders = [lesson.order for lesson in self.lessons]
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError(
                f"{self.id}: lesson orders must be 1..n with no gaps, got {sorted(orders)}"
            )

        known = {concept.id for concept in self.concepts}
        for lesson in self.lessons:
            if lesson.module_id != self.id:
                raise ValueError(f"{lesson.id}: module_id does not match {self.id}")
            if lesson.concept_id not in known:
                raise ValueError(
                    f"{lesson.id}: concept {lesson.concept_id!r} is not defined in "
                    f"this module"
                )

        for concept in self.concepts:
            for prerequisite in concept.prerequisites:
                # Only intra-module prerequisites are checked here.
                if prerequisite in known and prerequisite == concept.id:
                    raise ValueError(f"{concept.id}: depends on itself")
        return self

    def lessons_in_order(self) -> list[Lesson]:
        return sorted(self.lessons, key=lambda lesson: lesson.order)


class Curriculum:
    """Every module, loaded and cross-checked."""

    def __init__(self, modules: list[Module]) -> None:
        self.modules = sorted(modules, key=lambda module: module.order)
        self.by_id = {module.id: module for module in self.modules}
        self.concepts = {
            concept.id: concept for module in self.modules for concept in module.concepts
        }
        self.lessons = {
            lesson.id: lesson for module in self.modules for lesson in module.lessons
        }
        self.validate_prerequisites()
        self.validate_every_concept_is_taught()

    def validate_every_concept_is_taught(self) -> None:
        """A required concept with no lesson can never be mastered, and locks the graph.

        THE BUG THIS EXISTS FOR was caught in review rather than by a test,
        which is the whole argument for the check.

        A draft of the L4/L5 patch added `scarcity` as a concept node so that
        `need` could depend on it -- the client's own material for the youngest
        band teaches scarcity first, and it is the *why* behind needs-versus-
        wants -- and authored no lesson for it. Mastery is evidence, evidence
        comes from a lesson's check questions, so:

            scarcity  no lesson      -> never mastered
            need      needs scarcity -> never opens
            budget    needs need     -> never opens

        Half of module one, closed to every learner in every band, from one
        missing lesson and no error message anywhere.

        `validate_prerequisites` could not have caught it: every prerequisite
        named a concept that genuinely existed. The gap runs the other way -- a
        concept that exists, is required, and is never taught -- which is why
        this is a second check rather than another clause in the first.

        An untaught concept is legal when NOTHING requires it. A vocabulary node
        used for retrieval and depended on by nobody locks no path, so it is not
        this bug and is not refused here.
        """
        taught = {lesson.concept_id for module in self.modules for lesson in module.lessons}
        required: dict[str, list[str]] = {}
        for concept in self.concepts.values():
            for prerequisite in concept.prerequisites:
                required.setdefault(prerequisite, []).append(concept.id)

        orphaned = sorted(cid for cid in required if cid not in taught)
        if orphaned:
            detail = "; ".join(
                f"{cid!r} is required by {sorted(required[cid])} and no lesson teaches it"
                for cid in orphaned
            )
            raise ValueError(
                f"{detail}. A concept with no lesson can never be mastered, so "
                "everything downstream of it is unreachable. Author the lesson, "
                "or remove the prerequisite."
            )

    def validate_prerequisites(self) -> None:
        """Every prerequisite exists and is not above the band that needs it."""
        for concept in self.concepts.values():
            for prerequisite in concept.prerequisites:
                required = self.concepts.get(prerequisite)
                if required is None:
                    raise ValueError(
                        f"{concept.id}: prerequisite {prerequisite!r} is not defined "
                        "in any module"
                    )
                if band_index(required.band_min) > band_index(concept.band_min):
                    raise ValueError(
                        f"{concept.id} is available from {concept.band_min} but "
                        f"requires {prerequisite}, which starts at "
                        f"{required.band_min}"
                    )

    def lessons_for_band(self, band: str) -> list[Lesson]:
        """Every lesson a learner in this band may be given, in course order."""
        out: list[Lesson] = []
        for module in self.modules:
            if not band_index(module.band_min) <= band_index(band) <= band_index(
                module.band_max
            ):
                continue
            for lesson in module.lessons_in_order():
                concept = self.concepts[lesson.concept_id]
                if (
                    band_index(concept.band_min)
                    <= band_index(band)
                    <= band_index(concept.band_max)
                ):
                    out.append(lesson)
        return out

    def first_lesson_for(self, band: str) -> Lesson | None:
        lessons = self.lessons_for_band(band)
        return lessons[0] if lessons else None


def load_module(path: Path) -> Module:
    """One YAML file into a validated `Module`."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path.name} is not valid YAML: {error}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} does not contain a module object")
    try:
        return Module.model_validate(data)
    except Exception as error:
        raise ValueError(f"{path.name}: {error}") from None


_cache: Curriculum | None = None


def load_all(directory: Path | None = None, *, refresh: bool = False) -> Curriculum:
    """Every module in the content directory, validated together."""
    global _cache
    if _cache is not None and not refresh and directory is None:
        return _cache

    root = directory or CONTENT_DIR
    files = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not files:
        raise ValueError(
            f"No curriculum files in {root}. The learning agent has nothing to "
            "teach; refusing to pretend otherwise."
        )

    curriculum = Curriculum([load_module(path) for path in files])
    logger.info(
        "Curriculum loaded: %d module(s), %d lesson(s), %d concept(s).",
        len(curriculum.modules),
        len(curriculum.lessons),
        len(curriculum.concepts),
    )
    if directory is None:
        _cache = curriculum
    return curriculum
