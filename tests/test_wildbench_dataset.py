"""WildBench dataset normalization tests."""

import numpy as np
import pandas as pd
import pytest

from judgearena.datasets.wildbench import normalize_wildbench


def test_normalize_wildbench_preserves_multiturn_context():
    raw = pd.DataFrame(
        {
            "session_id": ["session-1"],
            "conversation_input": [
                np.array(
                    [
                        {"role": "user", "content": "First question"},
                        {"role": "assistant", "content": "First answer"},
                        {"role": "user", "content": "Follow-up"},
                    ],
                    dtype=object,
                )
            ],
            "checklist": [np.array(["Correct?", "Clear?"], dtype=object)],
            "primary_tag": ["Math"],
            "secondary_tags": [np.array(["Reasoning"], dtype=object)],
        }
    )

    row = normalize_wildbench(raw).iloc[0]

    assert row["instruction_index"] == "session-1"
    assert row["instruction"] == "Follow-up"
    assert row["history"] == "USER: First question\n\nASSISTANT: First answer\n\n"
    assert row["checklist"] == ["Correct?", "Clear?"]
    assert row["task_categories"] == [
        "Math & Data Analysis",
        "Planning & Reasoning",
    ]


def test_normalize_wildbench_rejects_non_user_final_turn():
    raw = pd.DataFrame(
        {
            "session_id": ["session-1"],
            "conversation_input": [[{"role": "assistant", "content": "Unexpected"}]],
            "checklist": [["Correct?"]],
            "primary_tag": ["Math"],
            "secondary_tags": [[]],
        }
    )

    with pytest.raises(ValueError, match="must end with a user turn"):
        normalize_wildbench(raw)
