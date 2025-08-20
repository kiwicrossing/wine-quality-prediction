from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from model.data_loading import WineDataLoader


class WineQualityModel:
    """
    A class to build and compile a neural network model for wine quality prediction.
    """

    def __init__(self):
        self.model = None

    def build_model(self):
        """
        Builds and compiles the neural network model.
        """
        model = Sequential()

        model.add(Dense(12, activation="relu", input_dim=12))
        model.add(Dense(9, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model = model
        return model

    def train_model(self, X_train, y_train):
        """
        Trains the compiled model using the provided training data.
        """
        if self.model is None:
            raise ValueError("Model has not been built.")

        self.model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1)

    def test_model(self, X_test):
        """
        Test the trained model on real data.
        """
        y_pred = self.model.predict(X_test)

        y_pred_labels = (y_pred >= 0.5).astype(int)

        for prediction in y_pred_labels[:12]:
            wine_type = "Red wine" if prediction == 1 else "White wine"
            print(f"Prediction: {wine_type}")
