import pytest

from judgearena.prompts.criteria import (
    SCALE_MAX,
    SCALE_MIN,
    Criterion,
    load_criteria_from_file,
    resolve_criteria,
)
from judgearena.prompts.registry import resolve_judge_prompt


def test_get_default_criteria():
    criteria = resolve_criteria()
    assert len(criteria) > 0
    assert [criterion.name for criterion in criteria] == [
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
    assert [criterion.name for criterion in criteria] == ["clarity", "correctness"]
    assert resolve_criteria(path) == criteria


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

    assert [criterion.name for criterion in criteria_yaml] == ["overall"]
    assert [criterion.name for criterion in criteria_yml] == ["overall"]


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


@pytest.fixture
def criteria_file(tmp_path):
    path = tmp_path / "criteria.yaml"
    path.write_text("""criteria:
  - name: clarity
    description: 'Explain {concept} using {"example": "value"} and {criteria_output_lines}.'
  - name: correctness
    description: Be correct.
""")
    return path


@pytest.mark.parametrize(
    "output",
    [
        "clarity: A=8, B=4",  # missing criterion
        "clarity: A=8, B=4\ncorrectness: A=6, B=8\n**Clarity**: A=6, B=8",  # duplicate
        "clarity: A=8, B=4\ncorrectness: A=0, B=8",  # outside the scale
        "clarity: A=8, B=4\ncorrectness: A=6, B=10.5",
        "clarity: A=8, B=4\ncorrectness: A=6.5oops, B=8",  # malformed number
        "clarity: A=8, B=4\ncorrectness: A=6, B=8\nextra: A=9, B=9",
    ],
)
def test_criteria_parser_rejects_incomplete_or_invalid_scores(criteria_file, output):
    prompt = resolve_judge_prompt(
        preset="criteria", options={"criteria_file": criteria_file}
    )
    assert prompt.parser.parse_result(output) is None


@pytest.mark.parametrize(
    "output",
    [
        "clarity: A=8, B=4\ncorrectness: A=6, B=8",
        "**Clarity**: A=8, B=4\n**Correctness**: A=6, B=8",
        "- clarity: A=8, B=4\n* correctness: A=6, B=8",
        "1. clarity: A=8, B=4\n2. correctness: A=6, B=8",
        "Assessment:\nclarity: A=8.0, B=4.5\ncorrectness: A=6.0, B=7.5\nA is clearer.",
        "clarity: A=8, B=4 — clearer.\ncorrectness: A=6, B=8 (more accurate).",
    ],
)
def test_criteria_parser_retains_scores_despite_formatting(criteria_file, output):
    parser = resolve_judge_prompt(
        preset="criteria", options={"criteria_file": criteria_file}
    ).parser
    result = parser.parse_result(output)
    assert result.scores == {"A": 7, "B": 6}
    assert set(result.details["criteria"]) == {"clarity", "correctness"}
    assert result.preference == pytest.approx(parser.preference_from_scores(7, 6))
