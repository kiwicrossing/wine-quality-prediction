import os
import pytest
import pandas as pd
import sqlite3
import warnings
from ..sql_analytics import WineSQLAnalytics

# Setup test database and table for testing
TEST_DB_PATH = "model/tests/test_wine_data.db"
TEST_TABLE_NAME = "wines"


@pytest.fixture(scope="module")
def setup_database():
    # Create a test database and populate it with sample data
    conn = sqlite3.connect(TEST_DB_PATH)
    df = pd.DataFrame(
        {
            "fixed_acidity": [7.4, 7.8, 7.8],
            "volatile_acidity": [0.70, 0.88, 0.76],
            "citric_acid": [0.00, 0.00, 0.04],
            "residual_sugar": [1.9, 2.6, 2.3],
            "chlorides": [0.076, 0.098, 0.092],
            "free_sulfur_dioxide": [11.0, 25.0, 15.0],
            "total_sulfur_dioxide": [34.0, 67.0, 54.0],
            "density": [0.9978, 0.9968, 0.9970],
            "pH": [3.51, 3.20, 3.26],
            "sulphates": [0.56, 0.68, 0.65],
            "alcohol": [9.4, 9.8, 10.0],
            "quality": [5, 5, 5],
            "type": ["red", "red", "red"],
        }
    )
    df.to_sql(TEST_TABLE_NAME, conn, index=False, if_exists="replace")
    conn.close()
    yield
    os.remove(TEST_DB_PATH)


def test_feature_correlation_with_quality(setup_database):
    analytics = WineSQLAnalytics(db_path=TEST_DB_PATH, table_name=TEST_TABLE_NAME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_df = analytics.feature_correlation_with_quality()
    assert isinstance(corr_df, pd.DataFrame)
    assert "Feature" in corr_df.columns
    assert "CorrelationWithQuality" in corr_df.columns
    assert not corr_df.empty


def test_save_correlation_table_csv(tmp_path, setup_database):
    analytics = WineSQLAnalytics(db_path=TEST_DB_PATH, table_name=TEST_TABLE_NAME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_df = analytics.feature_correlation_with_quality()
    save_path = tmp_path / "feature_correlations.csv"
    analytics.save_correlation_table_csv(corr_df, save_path=str(save_path))
    assert os.path.exists(save_path)
    saved_df = pd.read_csv(save_path)
    assert list(saved_df.columns) == ["Feature", "CorrelationWithQuality"]


def test_save_correlation_table_png(tmp_path, setup_database):
    analytics = WineSQLAnalytics(db_path=TEST_DB_PATH, table_name=TEST_TABLE_NAME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_df = analytics.feature_correlation_with_quality()
    save_path = tmp_path / "feature_correlations.png"
    analytics.save_correlation_table_png(corr_df, save_path=str(save_path))
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
