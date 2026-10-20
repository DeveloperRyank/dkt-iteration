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
