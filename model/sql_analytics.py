import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

class WineSQLAnalytics:
    def __init__(self, db_path="output/database_info/wine_data.db", table_name="wines"):
        self.db_path = db_path
        self.table_name = table_name

    def feature_correlation_with_quality(self):
        """
        Returns a DataFrame with the correlation of each chemical feature with wine quality.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
        conn.close()
        # Exclude non-chemical columns
        chemical_cols = [col for col in df.columns if col not in ("quality", "type")]
        corrs = {}
        for col in chemical_cols:
            corrs[col] = df[col].corr(df["quality"])
        corr_df = pd.DataFrame(list(corrs.items()), columns=["Feature", "CorrelationWithQuality"])
        corr_df = corr_df.sort_values("CorrelationWithQuality", key=abs, ascending=False)
        return corr_df

    def save_correlation_table_csv(self, corr_df, save_path="output/database_info/feature_correlations.csv"):
        """
        Saves the correlation DataFrame as a CSV file in output/database_info.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        corr_df.to_csv(save_path, index=False)
        print(f"Feature correlation CSV saved to {save_path}")

    def save_correlation_table_png(self, corr_df, save_path="output/feature_correlations.png"):
        """
        Saves the correlation DataFrame as a PNG table in output/.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(min(1 + 0.8 * len(corr_df.columns), 8), min(1 + 0.5 * len(corr_df), 12)))
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        # Format correlation values to 4 decimal places
        display_df = corr_df.copy()
        display_df["CorrelationWithQuality"] = display_df["CorrelationWithQuality"].round(4)

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            rowLabels=None,  # Hide row indices
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # Auto width for both columns
        col_labels = list(display_df.columns)
        col_indices = [col_labels.index("Feature"), col_labels.index("CorrelationWithQuality")]
        table.auto_set_column_width(col_indices)

        fig.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Feature correlation table saved to {save_path}")