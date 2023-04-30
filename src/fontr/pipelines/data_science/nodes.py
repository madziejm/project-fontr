"""todo nuke this file
"""

import logging
from typing import Dict, Tuple

# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calculates and logs the coefficient of determination.

    Args:
        model: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    # y_pred = model.predict(X_test)
    # score = r2_score(y_test, y_pred)
    # logger = logging.getLogger(__name__)
    # logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
