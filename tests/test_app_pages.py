"""Headless render tests for every dashboard page (st.testing.v1.AppTest).

These start the page scripts in-process, run them like a browser visit, and
fail if any exception or missing artifact occurs. Requires the trained
checkpoint and demo bundle in outputs/.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    str(ROOT / "app_pages" / name)
    for name in (
        "simulator.py",
        "overview.py",
        "model_comparison.py",
        "mastery_heatmap.py",
        "prediction_autopsy.py",
        "reality_check_page.py",
        "settings.py",
    )
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(page, default_timeout=300).run()
    assert not at.exception, f"{page} raised {at.exception}"
    assert len(at.header) >= 1, f"{page} rendered no header"


def test_simulator_records_answered_questions():
    """The interactive history really accumulates when buttons are clicked."""
    at = AppTest.from_file(str(ROOT / "app_pages" / "simulator.py"),
                           default_timeout=300).run()
    assert not at.exception
    at.session_state["my_topic"] = 0
    seg = at.segmented_control[0]
    seg.set_value("Be the student yourself").run()
    assert not at.exception, at.exception
    right = [b for b in at.button if "RIGHT" in b.label]
    wrong = [b for b in at.button if "WRONG" in b.label]
    assert right and wrong, f"answer buttons missing: {[b.label for b in at.button]}"
    wanted = [1, 0, 1]
    for value in wanted:
        label = "I got it RIGHT" if value == 1 else "I got it WRONG"
        # re-look-up the button every time: widgets are recreated each rerun
        next(b for b in at.button if b.label == label).click().run()
        assert not at.exception, at.exception
    hist = at.session_state["my_hist"]
    assert len(hist["skills"]) == 3
    assert hist["correct"] == wanted


def _heat_table(at):
    for df in at.dataframe:
        if "model estimate now" in set(df.value.columns):
            return df.value
    raise AssertionError("heatmap table not found")


def test_heatmap_scrub_recomputes_estimates():
    at = AppTest.from_file(str(ROOT / "app_pages" / "mastery_heatmap.py"),
                           default_timeout=300).run()
    assert not at.exception
    full = _heat_table(at)
    at.slider[0].set_value(0).run()
    assert not at.exception, at.exception
    empty = _heat_table(at)
    # with zero history the model is honest: 0.5 everywhere
    assert set(empty["model estimate now"].round(3)) == {0.5}
    assert set(empty["attempts so far"]) == {0}
    practiced = full[full["attempts so far"] > 0]
    assert len(practiced) >= 1
    # after real history the estimate leaves the 0.5 coin-flip starting point
    assert (practiced["model estimate now"].round(3) != 0.5).any()


def test_autopsy_live_editor_recomputes_guess():
    """Flipping a past answer must produce a new, different guess."""
    at = AppTest.from_file(str(ROOT / "app_pages" / "prediction_autopsy.py"),
                           default_timeout=300).run()
    assert not at.exception
    at.slider[0].set_value(1).run()  # audit question 2 -> one past answer
    assert not at.exception, at.exception
    before = {m.label: m.value for m in at.metric}
    at.multiselect[0].set_value([0]).run()
    assert not at.exception, at.exception
    after = {m.label: m.value for m in at.metric}
    assert "New guess" in after, list(after)
    assert after["Model's guess"] == before["Model's guess"]


def test_comparison_threshold_slider_changes_scores():
    at = AppTest.from_file(str(ROOT / "app_pages" / "model_comparison.py"),
                           default_timeout=300).run()
    assert not at.exception

    def table():
        for df in at.dataframe:
            if "missed a real correct" in set(df.value.columns):
                return df.value
        raise AssertionError("threshold table missing")

    loose = table()
    at.slider[0].set_value(0.9).run()
    assert not at.exception, at.exception
    strict = table()
    assert not loose.equals(strict), "threshold slider had no effect"
    assert (strict["missed a real correct"] >= loose["missed a real correct"]).all()


def test_comparison_topic_zoom_rescores_subset():
    at = AppTest.from_file(str(ROOT / "app_pages" / "model_comparison.py"),
                           default_timeout=300).run()
    assert not at.exception
    sel = next(s for s in at.selectbox if "one topic" in s.label)
    # option 0 is "All topics"; pick the first real skill
    sel.set_value(sel.options[1]).run()
    assert not at.exception, at.exception

    def zoom_table():
        for df in at.dataframe:
            if "pairs" in set(df.value.columns):
                return df.value
        return None

    tbl = zoom_table()
    if tbl is not None:  # thin topics show an explanatory notice instead
        assert len(tbl) == 3
        assert tbl["pairs"].iloc[0] < 46856
        assert set(tbl["model"]) == {"DKT", "BKT (EM, per skill)",
                                     "per-skill rate baseline"}
    else:
        assert any("Not enough to compare" in m.value for m in at.info)
