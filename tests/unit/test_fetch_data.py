"""Dataset fetching: cache, mirror download and the offline synthetic fallback."""

import pandas as pd
import pytest

from app.ml import fetch_data


def test_synthetic_dataset_matches_real_schema():
    df = fetch_data._generate_synthetic(200)
    assert set(df.columns) == fetch_data.EXPECTED_COLUMNS
    assert len(df) == 200 and set(df["Churn"]) <= {"Yes", "No"}
    assert 0.05 < (df["Churn"] == "Yes").mean() < 0.7  # has both classes


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_data, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fetch_data, "RAW_PATH", str(tmp_path / "telco.csv"))
    return tmp_path


def test_falls_back_to_synthetic_when_offline(isolated, monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(fetch_data.requests, "get", boom)
    df = fetch_data.fetch_telco_data(force=True)
    assert len(df) == 7043 and (isolated / "telco.csv").exists()


def test_uses_cached_file_without_network(isolated, monkeypatch):
    fetch_data._generate_synthetic(50).to_csv(isolated / "telco.csv", index=False)
    monkeypatch.setattr(fetch_data.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    assert len(fetch_data.fetch_telco_data()) == 50


def test_download_success_is_cached(isolated, monkeypatch):
    csv_bytes = fetch_data._generate_synthetic(30).to_csv(index=False).encode()

    class Resp:
        content = csv_bytes

        def raise_for_status(self):
            pass

    monkeypatch.setattr(fetch_data.requests, "get", lambda *a, **k: Resp())
    df = fetch_data.fetch_telco_data(force=True)
    assert isinstance(df, pd.DataFrame) and len(df) == 30


def test_bad_schema_download_is_rejected(isolated, monkeypatch):
    class Resp:
        content = b"a,b\n1,2\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(fetch_data.requests, "get", lambda *a, **k: Resp())
    assert len(fetch_data.fetch_telco_data(force=True)) == 7043  # fell through to synthetic
