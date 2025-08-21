import pytest
import numpy as np
import os
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

from ..neural_network import WineQualityModel


@pytest.fixture
def mock_model():
    model = WineQualityModel()
    model.history = MagicMock()
    model.history.history = {
        "loss": [0.8, 0.6, 0.4],
        "accuracy": [0.5, 0.7, 0.9],
        "val_loss": [0.9, 0.7, 0.5],
        "val_accuracy": [0.4, 0.6, 0.8],
    }
    return model


def test_build_model_creates_keras_model():
    model_wrapper = WineQualityModel()
    model = model_wrapper.build_model()
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    # Check input and output shapes
    assert model.input_shape[-1] == 12
    assert model.output_shape[-1] == 1


def test_train_model_raises_if_not_built():
    model_wrapper = WineQualityModel()
    X_train = np.random.rand(4, 12)
    y_train = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        model_wrapper.train_model(X_train, y_train)


def test_train_model_runs(monkeypatch):
    model_wrapper = WineQualityModel()
    model = model_wrapper.build_model()
    X_train = np.random.rand(4, 12)
    y_train = np.array([0, 1, 0, 1])

    # Patch fit to avoid actual training
    called = {}

    def fake_fit(X, y, epochs, batch_size, verbose, **kwargs):
        called["fit"] = True
        called["validation_data"] = kwargs.get("validation_data", None)
        return None

    monkeypatch.setattr(model, "fit", fake_fit)
    model_wrapper.train_model(X_train, y_train)
    assert called.get("fit", False)


def test_test_model_predicts_and_prints(monkeypatch, capsys):
    model_wrapper = WineQualityModel()
    model = model_wrapper.build_model()
    X_test = np.random.rand(3, 12)

    # Patch predict to return fixed values
    monkeypatch.setattr(model, "predict", lambda X: np.array([[1], [0], [1]]))
    model_wrapper.test_model(X_test)
    captured = capsys.readouterr()
    assert "Prediction: Red wine" in captured.out
    assert "Prediction: White wine" in captured.out


def test_plot_training_history(tmp_path, mock_model):
    save_dir = tmp_path
    mock_model.plot_training_history(save_dir=str(save_dir))
    loss_plot = save_dir / "loss_per_epoch.png"
    acc_plot = save_dir / "accuracy_per_epoch.png"
    assert loss_plot.exists()
    assert acc_plot.exists()
    assert os.path.getsize(loss_plot) > 0
    assert os.path.getsize(acc_plot) > 0


def test_save_training_history_table_plot(tmp_path, mock_model):
    save_dir = tmp_path
    mock_model.save_training_history_table_plot(save_dir=str(save_dir))
    table_plot = save_dir / "training_history_table.png"
    assert table_plot.exists()
    assert os.path.getsize(table_plot) > 0
