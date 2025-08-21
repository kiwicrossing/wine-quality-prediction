import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import os

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


class WineDataLoader:
    """
    A class to load, preprocess, and visualize wine quality data.
    """

    def __init__(self, red_path="dataset/winequality-red.csv", white_path="dataset/winequality-white.csv"):
        self.red_path = red_path
        self.white_path = white_path
        self.wines = None

    def load_data(self):
        """
        Loads red and white wine datasets from local CSV files, adds type labels, merges them, and removes missing values.
        """
        red = pd.read_csv(self.red_path, sep=";")
        white = pd.read_csv(self.white_path, sep=";")

        red["type"] = 1
        white["type"] = 0

        self.wines = pd.concat([red, white], ignore_index=True)
        self.wines.dropna(inplace=True)

    def get_data(self):
        """
        Returns the combined wine dataset, loading it if not already loaded.
        """
        if self.wines is None:
            self.load_data()
        return self.wines

    def plot_alcohol_distribution(self, save_path="output/alcohol_distribution.png"):
        """
        Plots histograms of alcohol content for red and white wines side by side.
        Saves the plot to a file if running in a non-interactive environment.
        """
        if self.wines is None:
            self.load_data()

        red_alcohol = self.wines[self.wines["type"] == 1].alcohol
        white_alcohol = self.wines[self.wines["type"] == 0].alcohol

        red_counts, _ = np.histogram(red_alcohol, bins=10, range=(8, 15))
        white_counts, _ = np.histogram(white_alcohol, bins=10, range=(8, 15))
        max_y = max(max(red_counts), max(white_counts)) + 50

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(
            red_alcohol,
            bins=10,
            range=(8, 15),
            facecolor="red",
            edgecolor="black",
            lw=0.5,
            alpha=0.5,
            label="Red wine",
        )
        ax[1].hist(
            white_alcohol,
            bins=10,
            range=(8, 15),
            facecolor="whitesmoke",
            edgecolor="black",
            lw=0.5,
            alpha=0.5,
            label="White wine",
        )

        for a in ax:
            a.set_ylim([0, max_y])
            a.set_xlim([8, 15])
            a.set_xlabel("Alcohol in % Vol")
            a.set_ylabel("Frequency")

        ax[0].set_title("Alcohol Content in Red Wine")
        ax[1].set_title("Alcohol Content in White Wine")

        fig.suptitle("Distribution of Alcohol by Wine Type")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    def split_data(self):
        """
        Splits the wine dataset into training and testing sets.
        Returns X_train, X_test, y_train, y_test.
        """
        if self.wines is None:
            self.load_data()

        X = self.wines.iloc[:, :-1]
        y = self.wines["type"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.34, random_state=45
        )

        return X_train, X_test, y_train, y_test

    def save_to_sqlite(self, db_path="output/database_info/wine_data.db", table_name="wines"):
        """
        Saves the combined wine DataFrame to a SQLite database.
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if self.wines is None:
            self.load_data()
        conn = sqlite3.connect(db_path)
        self.wines.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        print(f"Data saved to {db_path} in table '{table_name}'")
