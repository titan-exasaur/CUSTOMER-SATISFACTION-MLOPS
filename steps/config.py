from zenml.steps import base_parameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    model_name : str = "LinearRegression"
    