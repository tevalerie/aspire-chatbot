"""Zion is one voice with two rungs, and the older rung is not the younger one.

`persona_voice` wins over the band map -- which was right until the reader was
a seventeen-year-old being taught in orion's 13-15 voice. A voice declares the
band it was authored AT; a reader above it takes the band map's older words.
"""

from app.curriculum.schema import load_all


def _lesson():
    return load_all(refresh=False).modules[0].lessons[0]


class TestTheSecondRung:
    def test_13_15_gets_the_voice(self):
        lesson = _lesson()
        assert lesson.teach_for("13-15", "orion") == list(
            lesson.persona_voice["orion"].teach_points
        )

    def test_16_18_outgrows_it(self):
        lesson = _lesson()
        assert lesson.teach_for("16-18", "orion") == list(
            lesson.teach_points["16-18"]
        )

    def test_adult_voices_are_not_outgrown(self):
        """Guest/Imani/Azuri are authored at adult; adult readers keep them."""
        lesson = _lesson()
        assert lesson.teach_for("adult", "guest") == list(
            lesson.persona_voice["guest"].teach_points
        )
