from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import pandas as pd
import os

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
        model.add(Input(shape=(12,)))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(9, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model = model
        return model

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=9, batch_size=32):
        """
        Trains the compiled model using the provided training data.
        Optionally accepts validation data for test accuracy/loss per epoch.
        """
        if self.model is None:
            raise ValueError("Model has not been built.")

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        )
        self.history = history
        return history

    def test_model(self, X_test):
        """
        Test the trained model on real data.
        """
        y_pred = self.model.predict(X_test)

        y_pred_labels = (y_pred >= 0.5).astype(int)

        for prediction in y_pred_labels[:12]:
            wine_type = "Red wine" if prediction == 1 else "White wine"
            print(f"Prediction: {wine_type}")

    def plot_training_history(self, save_dir="output"):
        """
        Plots the training and validation accuracy and loss for each epoch and saves the plots as PNG files.
        """
        if not hasattr(self, "history"):
            raise ValueError("No training history found. Train the model first.")

        os.makedirs(save_dir, exist_ok=True)
        history = self.history

        # Plot loss
        plt.figure(figsize=(6, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_per_epoch.png")
        plt.close()

        # Plot accuracy
        plt.figure(figsize=(6, 5))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/accuracy_per_epoch.png")
        plt.close()

    def save_training_history_table_plot(self, save_dir="output"):
        """
        Plots the training and validation accuracy and loss for each epoch as a matplotlib table and saves it as a PNG.
        """
        if not hasattr(self, "history"):
            raise ValueError("No training history found. Train the model first.")

        os.makedirs(save_dir, exist_ok=True)
        history = self.history.history
        epochs = list(range(1, len(history["loss"]) + 1))
        data = {
            "Epoch": epochs,
            "Train Loss": history["loss"],
            "Train Accuracy": history["accuracy"],
        }
        if "val_loss" in history:
            data["Val Loss"] = history["val_loss"]
        if "val_accuracy" in history:
            data["Val Accuracy"] = history["val_accuracy"]

        df = pd.DataFrame(data)
        df["Epoch"] = df["Epoch"].astype(int)  # Ensure epochs are integers

        fig, ax = plt.subplots(figsize=(min(1 + 0.8 * len(df.columns), 10), min(1 + 0.5 * len(df), 20)))
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        # Format cell text: Epoch as int, others as rounded floats
        cell_text = []
        for i, row in df.iterrows():
            formatted_row = [f"{int(row['Epoch'])}"]  # Epoch as int string
            for col in df.columns[1:]:
                formatted_row.append(f"{row[col]:.4f}")
            cell_text.append(formatted_row)

        table = ax.table(cellText=cell_text, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # Ensure "Train Accuracy" and "Val Accuracy" columns are wide enough
        col_labels = list(df.columns)
        col_indices = []
        for col_name in ["Train Accuracy", "Val Accuracy"]:
            if col_name in col_labels:
                col_indices.append(col_labels.index(col_name))
        if col_indices:
            table.auto_set_column_width(col_indices)

        fig.tight_layout()
        plt.savefig(f"{save_dir}/training_history_table.png")
        plt.close()
        print(f"Training history table saved to {save_dir}/training_history_table.png")
