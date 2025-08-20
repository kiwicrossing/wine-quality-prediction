from model.data_loading import WineDataLoader
from model.neural_network import WineQualityModel


def main():
    # Load and split data
    loader = WineDataLoader()
    loader.plot_alcohol_distribution()
    X_train, X_test, y_train, y_test = loader.split_data()

    # Build and train model
    nn = WineQualityModel()
    model = nn.build_model()
    nn.train_model(X_train, y_train)

    # Test the model on new data
    nn.test_model(X_test)


if __name__ == "__main__":
    main()
