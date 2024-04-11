import logging
from zenml import step
import pandas as pd

@step
def evaluate_model(df:pd.DataFrame)->None:
    """
    Evaluates the model on  test partition of ingested data

    Args:
        df : the ingested data
    """
    pass