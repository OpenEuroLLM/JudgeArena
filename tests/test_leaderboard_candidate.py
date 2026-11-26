from pathlib import Path

import pandas as pd

from judgearena.leaderboard.prepare import comparison_groups, reference_pool


def test_grouped_split_and_connected_pool():
    frame = pd.DataFrame(
        {
            "instruction": ["same", "other", "same", "separate"],
            "session": ["s1", "s1", "s2", "s3"],
            "model_a": ["a", "b", "a", "x"],
            "model_b": ["b", "c", "c", "y"],
            "reference_pref": [0.5] * 4,
        }
    )
    groups = comparison_groups(frame)
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2]
    assert groups.iloc[3] != groups.iloc[0]
    pool = reference_pool(frame, minimum=1)
    assert set(pool.model_a) | set(pool.model_b) == {"a", "b", "c"}


def test_create_from_one_config_and_replay(tmp_path, monkeypatch):
    import pytest
    import yaml

    from judgearena.cli import cli
    from judgearena.leaderboard.release import load_release
    from judgearena.models import PreparedModel
    from judgearena.tasks.definitions.lmarena import dataset

    rows = pd.DataFrame(
        {
            "question_id": range(60),
            "model_a": "a",
            "model_b": "b",
            "winner": "tie",
            "lang": ["en"] * 30 + ["de"] * 30,
            "evaluation_session_id": [f"s{i // 2}" for i in range(60)],
            **{
                f"conversation_{slot}": [
                    [
                        dict(role="user", content=f"Q{i}"),
                        dict(role="assistant", content=answer),
                    ]
                    for i in range(60)
                ]
                for slot, answer in (("a", "A"), ("b", "B"))
            },
        }
    )

    monkeypatch.setattr(dataset, "_read_release", lambda *_: rows)
    config = yaml.safe_load(
        Path("configs/leaderboard/lmarena-140k-v1.yaml").read_text()
    )
    config["judge"]["model"] = "Dummy/[[A>B]]"
    config["run"].update(
        store_root=str(tmp_path / "cache"), filters={"lang": ["en", "de"]}
    )
    for metric in config["protocol"]["scoring"]["metrics"]:
        metric["parameters"].update(baseline_model="a", n_bootstraps=10)
    config["leaderboard"].update(
        version="test",
        sampling=dict(
            minimum_model_comparisons=1,
            comparisons_per_opponent=2,
            max_reference_comparisons=30,
        ),
    )
    path = tmp_path / "config.yaml"
    for labels in ("human", "judge"):
        config["leaderboard"]["labels"] = labels
        if labels == "judge":
            rows["winner"] = None  # Judge creation must not require human labels.
        path.write_text(yaml.safe_dump(config))
        output = tmp_path / labels
        command = [
            "leaderboard",
            "create",
            "--config",
            str(path),
            "--leaderboard.sampling.max_reference_comparisons",
            "20",
        ]
        cli([*command, "--output", str(output)])
        _, ratings = load_release(output)
        reference = pd.read_parquet(output / "reference_battles.parquet")
        panel = pd.read_parquet(output / "panel.parquet")
        heldout = pd.read_parquet(output / "heldout_battles.parquet")
        assert reference.groupby("lang").size().max() <= 20
        assert reference.groupby("session").size().eq(2).all()
        assert not set(reference.split_group) & set(heldout.split_group)
        assert set(panel.battle_id) <= set(reference.battle_id)
        assert panel.groupby(["lang", "opponent_model"]).size().eq(2).all()
        assert ratings[["ci_low", "ci_high"]].notna().all().all()
        with monkeypatch.context() as patch:
            patch.setattr(
                PreparedModel, "materialize", lambda *a, **k: pytest.fail("cache miss")
            )
            curated = tmp_path / f"{labels}-curated"
            patch.setattr(
                dataset,
                "_read_release",
                lambda *a: pytest.fail("candidate must bypass adapter"),
            )
            cli([*command, "--candidate", str(output), "--output", str(curated)])
        pd.testing.assert_frame_equal(
            heldout, pd.read_parquet(curated / "heldout_battles.parquet")
        )
        _, curated_ratings = load_release(curated)
        pd.testing.assert_frame_equal(ratings, curated_ratings)


def test_common_opponents():
    import pytest

    from judgearena.leaderboard.prepare import sample_battles
    from judgearena.leaderboard.settings import LeaderboardSettings

    battles = pd.DataFrame(
        [
            dict(
                battle_id=f"{lang}{i}",
                instruction=f"{lang}{i}",
                lang=lang,
                model_a="a",
                model_b=model,
                completion_a="A",
                completion_b="B",
                reference_pref=0.5,
            )
            for lang in ("en", "de")
            for i, model in enumerate(["b"] * 6 + [lang] * 6 + ["rare"])
        ]
    )
    settings = LeaderboardSettings(
        version="test",
        sampling=dict(
            validation_fraction=0,
            minimum_model_comparisons=2,
            comparisons_per_opponent=2,
        ),
    )

    def sample():
        return sample_battles(battles, settings, seed=42, baseline_models=[])

    references, panel, _ = sample()
    assert set(references.model_b) == {"b", "en", "de"}
    assert panel.groupby("lang").opponent_model.unique().map(list).tolist() == [
        ["a", "b"],
        ["a", "b"],
    ]
    settings.sampling.opponents = ["b", "a"]
    assert sample()[1].opponent_model.drop_duplicates().tolist() == ["b", "a"]
    settings.sampling.opponents = ["en"]
    with pytest.raises(ValueError, match="absent"):
        sample()
    settings.sampling.opponents = None
    battles.loc[battles.lang == "de", "model_a"] = "different"
    battles.loc[battles.lang == "de", "model_b"] += "-de"
    with pytest.raises(ValueError, match="No common eligible"):
        sample()
