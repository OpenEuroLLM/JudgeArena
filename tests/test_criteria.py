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
    assert resolve_criteria("not-a-builtin", path) == criteria


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
        "clarity: A=8, B=4\nclarity: A=6, B=8",  # duplicate
        "clarity: A=8, B=4\ncorrectness: A=0, B=8",  # outside the scale
        "clarity: A=8, B=4\ncorrectness: A=6, B=11",
        "clarity: A=8, B=4\ncorrectness: A=6.5, B=8",  # noninteger
        "clarity: A=8, B=4\ncorrectness: A=6, B=8\nextra: A=9, B=9",
    ],
)
def test_criteria_parser_rejects_incomplete_or_invalid_scores(criteria_file, output):
    prompt = resolve_judge_prompt(preset="criteria", criteria_file=criteria_file)
    assert prompt.parser.parse_result(output) is None
