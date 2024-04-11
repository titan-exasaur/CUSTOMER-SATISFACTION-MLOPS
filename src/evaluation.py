import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating our models
    """
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates the scores of the model
        
        Args:
            y_true : True Labels
            y_pred : Predicted Labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info(f"MSE : {mse}")
            return mse
        except Exception as e:
            logging.error("Error in Calculating MSE : {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that uses r2 score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating r2 score")
            r2 = r2_score(y_true,y_pred)
            logging.info(f"R2 Score : {r2}")
            return mse
        except r2 as e:
            logging.error("Error in Calculating r2 score : {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true,y_pred))
            logging.info(f"RMSE : {rmse}")
            return rmse
        except Exception as e:
            logging.error("Error in Calculating RMSE : {}".format(e))
            raise e