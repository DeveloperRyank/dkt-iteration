import numpy as np
import pandas as pd
import pytest
import torch

from model.dkt import DKT
from reality_check import (agreement_report, collect_with_positions,
                           make_template_csv, position_band_report,
                           validate_imported)

SKILLS = ["addition", "subtraction", "fractions"]


def valid_df():
    return pd.DataFrame({
        "student_index": [0, 1],
        "skill_name": ["addition", "fractions"],
        "question_id": ["q1", "q2"],
        "is_transfer": [0, 1],
        "correct": [1, 0],
    })


def test_validate_accepts_good_file():
    assert validate_imported(valid_df(), SKILLS, 5) == []


def test_validate_rejects_missing_column():
    df = valid_df().drop(columns=["correct"])
    problems = validate_imported(df, SKILLS, 5)
    assert any("correct" in p for p in problems)


def test_validate_rejects_unknown_skill():
    df = valid_df()
    df.loc[0, "skill_name"] = "astrochemistry"
    problems = validate_imported(df, SKILLS, 5)
    assert any("astrochemistry" in p for p in problems)


def test_validate_rejects_out_of_range_student():
    df = valid_df()
    df["student_index"] = [42, 1]
    assert validate_imported(df, SKILLS, 5) != []


def test_validate_rejects_bad_values():
    df = valid_df()
    df["is_transfer"] = ["maybe", "no"]
    assert validate_imported(df, SKILLS, 5) != []


def test_template_roundtrip():
    import io
    df = pd.read_csv(io.StringIO(make_template_csv(SKILLS)))
    assert list(df.columns) == ["student_index", "skill_name", "question_id",
                                "is_transfer", "correct"]


def test_agreement_report_separates_observed_and_predicted():
    torch.manual_seed(0)
    model = DKT(3, 8)
    seqs = [{"skills": [0, 1, 2, 0], "correct": [1, 0, 1, 1]},
            {"skills": [1, 2, 1], "correct": [0, 1, 1]}]
    rep = agreement_report(valid_df().iloc[:1], model, seqs,
                           {n: i for i, n in enumerate(SKILLS)})
    assert rep["n"] == 1
    row = rep["rows"][0]
    assert {"predicted", "correct", "transfer"} <= set(row)
    assert 0 <= row["predicted"] <= 1 and row["correct"] in (0, 1)
    s = rep["summary"][0]
    assert s["n"] == 1
    # observed and predicted are reported as separate fields, not merged
    assert "mean_predicted" in s and "observed_accuracy" in s


def test_position_bands_partition_eligible_pairs():
    torch.manual_seed(0)
    model = DKT(2, 4)
    seqs = [{"skills": list(np.random.default_rng(1).integers(0, 2, 30)),
             "correct": list(np.random.default_rng(2).integers(0, 2, 30))}
            for _ in range(6)]
    m = collect_with_positions(model, seqs)
    rows = position_band_report(m)
    total = sum(r["n"] for r in rows)
    assert total == int(m["seen"].sum())
