from model.data_loading import WineDataLoader
from model.neural_network import WineQualityModel
from model.sql_analytics import WineSQLAnalytics

def main():
    # Load and split data
    loader = WineDataLoader()
    loader.plot_alcohol_distribution()
    X_train, X_test, y_train, y_test = loader.split_data()

    # Save data to SQLite for analytics
    loader.save_to_sqlite()

    # Feature importance analysis using SQL
    analytics = WineSQLAnalytics()
    corr_df = analytics.feature_correlation_with_quality()
    print("\nFeature correlations with wine quality:")
    print(corr_df)
    analytics.save_correlation_table_csv(corr_df, save_path="output/database_info/feature_correlations.csv")
    analytics.save_correlation_table_png(corr_df, save_path="output/feature_correlations.png")

    # Build and train model
    nn = WineQualityModel()
    model = nn.build_model()
    nn.train_model(X_train, y_train, X_val=X_test, y_val=y_test)

    # Plot and save training history
    nn.plot_training_history(save_dir="output")
    nn.save_training_history_table_plot(save_dir="output")

    # Test the model on new data
    nn.test_model(X_test)


if __name__ == "__main__":
    main()
