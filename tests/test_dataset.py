from pathlib import Path

from nanochat import dataset

from tests.conftest import write_parquet_file


def test_list_parquet_files_filters_tmp_and_sorts(tmp_path):
    (tmp_path / "b.parquet").touch()
    (tmp_path / "a.parquet").touch()
    (tmp_path / "ignore.tmp").touch()
    (tmp_path / "c.parquet.tmp").touch()

    files = dataset.list_parquet_files(str(tmp_path))

    assert files == [
        str(tmp_path / "a.parquet"),
        str(tmp_path / "b.parquet"),
    ]


def test_parquets_iter_batched_respects_split_start_and_step(tmp_path):
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    write_parquet_file(train_path, [["train-rg0-a", "train-rg0-b"], ["train-rg1"], ["train-rg2"]])
    write_parquet_file(val_path, [["val-rg0"], ["val-rg1"]])

    train_batches = list(dataset.parquets_iter_batched("train", data_dir=str(tmp_path), start=1, step=2))
    val_batches = list(dataset.parquets_iter_batched("val", data_dir=str(tmp_path), start=0, step=1))

    assert train_batches == [["train-rg1"]]
    assert val_batches == [["val-rg0"], ["val-rg1"]]


def test_download_repo_snapshot_skips_existing_directory(tmp_path, monkeypatch):
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    (existing_dir / "marker.txt").write_text("ready", encoding="utf-8")
    called = []

    monkeypatch.setattr(dataset, "snapshot_download", lambda **kwargs: called.append(kwargs))

    dataset._download_repo_snapshot(1, 6, "pretrain", str(existing_dir), "owner/repo")

    assert called == []


def test_download_repo_snapshot_downloads_missing_directory(tmp_path, monkeypatch):
    target_dir = tmp_path / "missing"
    called = []

    def fake_snapshot_download(**kwargs):
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["local_dir"]) / "done.txt").write_text("ok", encoding="utf-8")
        called.append(kwargs)

    monkeypatch.setattr(dataset, "snapshot_download", fake_snapshot_download)

    dataset._download_repo_snapshot(2, 6, "eval", str(target_dir), "owner/eval")

    assert called == [
        {
            "repo_id": "owner/eval",
            "repo_type": "dataset",
            "local_dir": str(target_dir),
        }
    ]
    assert (target_dir / "done.txt").exists()
