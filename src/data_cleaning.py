import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data_path:str) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data_path: str) -> pd.DataFrame | pd.Series:
        """
        Pre-process data
        """
        try:
            #dropping unnecessary columns
            data = data.drop([
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_data",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],axis=1)
            
            # #dropping NaN values
            # data = data.dropna()

            #handling NaN values
            data['product_weight_g'].fillna(data['product_weight_g'].median(),inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(),inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(),inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(),inplace=True)
            data['review_comment_message'].fillna('No Review',inplace=True)

            #selecting columns that are numeric
            data = data.select_dtypes(include=[np.number])

            #droppping unnecessary columns from numeric columns
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop,axis=1)
            return data
        
        except Exception as e:
            logging.error("Error in preprocessing data : {}".format(e))
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test 
    """
    def handle_data(self, data_path: str) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(['review_score'], axis=1)#choosing features
            y = data['review_score']#choosing target

            X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=100,test_size=0.2)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data : {}".format(e))
            raise e
    
class DataCleaning:
    """
    class for cleaning and dividing data into train and test partitions
    """
    def __init__(self, data:pd.DataFrame, strategy:DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data : {e}")
            raise e
    
# if __name__ == "__main__":
#     data = pd.read_csv('/home/ak/Desktop/Desktop/NEW JOB/UPSKILL/MLOPS/data/olist_customers_dataset.csv')
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data()