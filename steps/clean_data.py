import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing import Annotated, Tuple

@step
def clean_df(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
    ]:
    """
    cleans the data and divides it into train and test

    Args:
        df : Raw Data
    
    Returns:
        X_train : Training data
        X_test : Testing Data
        y_train : Training Labels
        y_test : Testing Labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()


        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, process_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Complete")

    except Exception as e:
        logging.error(f"Error in cleaning data : {e}")
        raise e