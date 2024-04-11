import logging
from zenml import step
import pandas as pd
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Annotated, Tuple

@step
def evaluate_model(model:RegressorMixin,
                   X_test : pd.DataFrame,
                   y_test : pd.Series) -> Tuple[
                       Annotated[float, "r2_score"],
                       Annotated[float, "mse"],
                       Annotated[float, "rmse"]
                   ]:
    """
    Evaluates the model on  test partition of ingested data

    Args:
        df : the ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        return mse, rmse, r2

    except Exception as e:
        logging.error(f"Error in evaluating model : {e}")
        raise e