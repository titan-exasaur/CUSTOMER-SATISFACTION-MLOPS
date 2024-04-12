from zenml.client import Client
from pipelines.training_pipelines import train_pipeline

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/ak/Desktop/Desktop/NEW JOB/UPSKILL/MLOPS/data/olist_customers_dataset.csv")
