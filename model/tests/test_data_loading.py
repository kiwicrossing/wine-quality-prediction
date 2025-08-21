import pytest
import pandas as pd
import numpy as np
import os
import sqlite3
from unittest.mock import patch

from ..data_loading import WineDataLoader


@pytest.fixture
def sample_red_df():
    return pd.DataFrame(
        {
            "fixed acidity": [7.4, 7.8],
            "volatile acidity": [0.7, 0.88],
            "citric acid": [0, 0],
            "residual sugar": [1.9, 2.6],
            "chlorides": [0.076, 0.098],
            "free sulfur dioxide": [11, 25],
            "total sulfur dioxide": [34, 67],
            "density": [0.9978, 0.9968],
            "pH": [3.51, 3.2],
            "sulphates": [0.56, 0.68],
            "alcohol": [9.4, 9.8],
            "quality": [5, 5],
        }
    )


@pytest.fixture
def sample_white_df():
    return pd.DataFrame(
        {
            "fixed acidity": [7.0, 6.3],
            "volatile acidity": [0.27, 0.3],
            "citric acid": [0.36, 0.34],
            "residual sugar": [20.7, 1.6],
            "chlorides": [0.045, 0.049],
            "free sulfur dioxide": [45, 14],
            "total sulfur dioxide": [170, 132],
            "density": [1.001, 0.994],
            "pH": [3, 3.3],
            "sulphates": [0.45, 0.49],
            "alcohol": [8.8, 9.5],
            "quality": [6, 6],
        }
    )


@patch("pandas.read_csv")
def test_load_data_merges_and_labels(mock_read_csv, sample_red_df, sample_white_df):
    # Mock red and white wine data
    mock_read_csv.side_effect = [sample_red_df.copy(), sample_white_df.copy()]
    loader = WineDataLoader()
    loader.load_data()
    wines = loader.wines

    # Check shape and type column
    assert wines.shape[0] == 4
    assert "type" in wines.columns
    assert wines["type"].tolist().count(1) == 2
    assert wines["type"].tolist().count(0) == 2


def test_get_data_loads_if_none(monkeypatch, sample_red_df, sample_white_df):
    loader = WineDataLoader()

    # Patch load_data to set wines
    def fake_load():
        loader.wines = pd.concat(
            [sample_red_df.assign(type=1), sample_white_df.assign(type=0)],
            ignore_index=True,
        )

    monkeypatch.setattr(loader, "load_data", fake_load)
    wines = loader.get_data()
    assert isinstance(wines, pd.DataFrame)
    assert "type" in wines.columns


@patch("pandas.read_csv")
def test_plot_alcohol_distribution_saves_file(
    mock_read_csv, tmp_path, sample_red_df, sample_white_df
):
    mock_read_csv.side_effect = [sample_red_df.copy(), sample_white_df.copy()]
    loader = WineDataLoader()
    loader.load_data()
    save_path = tmp_path / "alcohol_dist.png"
    loader.plot_alcohol_distribution(save_path=str(save_path))
    assert save_path.exists()


@patch("pandas.read_csv")
def test_split_data_returns_correct_shapes(
    mock_read_csv, sample_red_df, sample_white_df
):
    mock_read_csv.side_effect = [sample_red_df.copy(), sample_white_df.copy()]
    loader = WineDataLoader()
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    # There are 4 samples, test_size=0.34 -> 2 test, 2 train (since 4*0.34=1.36, rounds up)
    assert X_train.shape[0] + X_test.shape[0] == 4
    assert y_train.shape[0] + y_test.shape[0] == 4
    assert set(X_train.columns) == set(sample_red_df.columns)
    assert set(X_test.columns) == set(sample_red_df.columns)
    assert y_train.isin([0, 1]).all()
    assert y_test.isin([0, 1]).all()


@patch("pandas.read_csv")
def test_save_to_sqlite(mock_read_csv, sample_red_df, sample_white_df, tmp_path):
    mock_read_csv.side_effect = [sample_red_df.copy(), sample_white_df.copy()]
    loader = WineDataLoader()
    loader.load_data()

    db_path = tmp_path / "wine_data.db"
    table_name = "wines"
    loader.save_to_sqlite(db_path=str(db_path), table_name=table_name)

    assert os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    assert not df.empty
    assert "type" in df.columns
    assert df.shape[0] == 4
