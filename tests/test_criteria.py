import pytest

from judgearena.criteria.defaults import CRITERIA_BY_NAME
from judgearena.criteria.io import load_criteria_from_file
from judgearena.criteria.schema import SCALE_MAX, SCALE_MIN, Criterion, criterion_names


def test_get_default_criteria():
    assert "clarity" in criterion_names(CRITERIA_BY_NAME["default"])


def test_load_criteria_from_file_supports_yaml(tmp_path):
    path = tmp_path / "criteria.yaml"
    path.write_text("criteria: [{name: clarity, description: Clarity}]\n")
    assert criterion_names(load_criteria_from_file(path)) == ["clarity"]


def test_load_criteria_from_file_rejects_unknown_extension(tmp_path):
    path = tmp_path / "criteria.txt"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported criteria file format"):
        load_criteria_from_file(path)


def test_criterion_rejects_out_of_range_score_reference():
    with pytest.raises(ValueError, match="outside the configured scale"):
        Criterion(
            name="clarity",
            description="Clarity",
            score_references={SCALE_MAX + 1: "Too high"},
        )


def test_criterion_normalizes_score_reference_keys():
    criterion = Criterion(
        name="clarity",
        description="Clarity",
        score_references={"1": "Poor", str(SCALE_MAX): "Great"},
    )
    assert criterion.score_references == {SCALE_MIN: "Poor", SCALE_MAX: "Great"}
