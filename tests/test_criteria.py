import pytest

from judgearena.prompts.criteria import (
    CRITERIA_BY_NAME,
    SCALE_MAX,
    SCALE_MIN,
    Criterion,
    criterion_names,
    load_criteria_from_file,
)


def test_get_default_criteria():
    criteria = CRITERIA_BY_NAME["default"]
    assert len(criteria) > 0
    assert criterion_names(criteria) == [
        "adherence",
        "helpfulness",
        "factuality",
        "completeness",
        "clarity",
        "fluency",
    ]


def test_load_criteria_from_file_supports_yaml(tmp_path):
    path = tmp_path / "criteria.yaml"
    path.write_text(
        """
name: yaml_criteria
criteria:
  - name: clarity
    description: Clarity
  - name: correctness
    description: Correctness
"""
    )

    criteria = load_criteria_from_file(path)
    assert criterion_names(criteria) == ["clarity", "correctness"]


def test_load_criteria_from_file_supports_yaml_and_yml(tmp_path):
    content = """
name: file_loader
criteria:
  - name: overall
    description: Overall quality
"""
    yaml_path = tmp_path / "criteria.yaml"
    yml_path = tmp_path / "criteria.yml"
    yaml_path.write_text(content)
    yml_path.write_text(content)

    criteria_yaml = load_criteria_from_file(yaml_path)
    criteria_yml = load_criteria_from_file(yml_path)

    assert criterion_names(criteria_yaml) == ["overall"]
    assert criterion_names(criteria_yml) == ["overall"]


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
