import judgearena.utils as utils


def test_download_all_dispatches_arena_hard_versions(monkeypatch, tmp_path):
    calls: list[tuple[str, str, object]] = []

    monkeypatch.setattr(utils, "data_root", tmp_path)
    monkeypatch.setattr(
        utils,
        "download_hf",
        lambda name, local_path: calls.append(("hf", name, local_path)),
    )
    monkeypatch.setattr(
        utils,
        "download_arena_hard",
        lambda dataset, local_tables_path: calls.append(
            ("arena", dataset, local_tables_path)
        ),
    )
    monkeypatch.setattr(
        utils,
        "snapshot_download",
        lambda **kwargs: calls.append(
            ("snapshot", kwargs["repo_id"], kwargs["local_dir"])
        ),
    )

    utils.download_all()

    tables_dir = tmp_path / "tables"
    assert calls[:4] == [
        ("hf", "alpaca-eval", tables_dir),
        ("arena", "arena-hard-v0.1", tables_dir),
        ("arena", "arena-hard-v2.0", tables_dir),
        ("hf", "m-arena-hard", tables_dir),
    ]
    assert calls[4] == (
        "snapshot",
        "geoalgo/multilingual-contexts-to-be-completed",
        tmp_path / "contexts",
    )
