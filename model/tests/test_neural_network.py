import pytest
import numpy as np

from ..neural_network import WineQualityModel

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
    def fake_fit(X, y, epochs, batch_size, verbose):
        called["fit"] = True
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