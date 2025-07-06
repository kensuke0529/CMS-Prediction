from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import pandas as pd
import numpy as np
import boto3
import os
from pathlib import Path

from sagemaker.image_uris import retrieve
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner
from sagemaker.inputs import TrainingInput

sagemaker_session = Session()
bucket = sagemaker_session.default_bucket()
prefix = 'XGBoost-Regressor'
key = 'XGBoost-Regressor'
role = 'arn:aws:iam::291480921130:role/service-role/AmazonSageMaker-ExecutionRole-20250617T212095'


def upload():
    base = Path().resolve()
    train_file = base / 'data' / 'train.csv'
    val_file = base / 'data' / 'val.csv'

    s3 = boto3.Session().resource('s3')

    train_key = os.path.join(prefix, 'train', train_file.name)
    val_key = os.path.join(prefix, 'val', val_file.name)

    s3.Bucket(bucket).Object(train_key).upload_file(str(train_file))
    s3.Bucket(bucket).Object(val_key).upload_file(str(val_file))

    s3_train_data = f's3://{bucket}/{train_key}'
    s3_val_data = f's3://{bucket}/{val_key}'
    output_location = f's3://{bucket}/{prefix}/output'

    print(f"Uploaded training data: {s3_train_data}")
    print(f"Uploaded validation data: {s3_val_data}")
    print(f"Training output will be saved to: {output_location}")

    return s3_train_data, s3_val_data, output_location

def train(s3_train_data, s3_val_data, output_location):
    container = retrieve("xgboost", boto3.Session().region_name, version="1.5-1")

    xgb_estimator = Estimator(
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",   
        output_path=output_location,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_run=900,      # 15 min
        max_wait=1800,    # 30 min 
    )

    hyperparameter_ranges = {
        "eta": ContinuousParameter(0.1, 0.3),       
        "max_depth": IntegerParameter(3, 5),        
        "num_round": IntegerParameter(5, 6)    
    }

    tuner = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name="validation:rmse",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[
            {
                "Name": "validation:rmse",
                "Regex": ".*\\[0\\]\\s+validation-rmse:([0-9\\.]+)"
            }
        ],
        max_jobs=2,       
        max_parallel_jobs=1,
        objective_type="Minimize", 
    )

    train_input = TrainingInput(s3_train_data, content_type="text/csv")
    val_input = TrainingInput(s3_val_data, content_type="text/csv")
    tuner.fit({"train": train_input, "validation": val_input})

    best_estimator = tuner.best_estimator()
    print("Best model artifact at:", best_estimator.model_data)

    df_all = tuner.analytics().dataframe()
    df_sorted = df_all.sort_values("FinalObjectiveValue", ascending=True)
    df_sorted[['TrainingJobName', 'FinalObjectiveValue']].head()

def main():
    s3_train_data, s3_val_data, output_location = upload()
    train(s3_train_data, s3_val_data, output_location)

if __name__ == "__main__":
    main()

"""
2025-07-06 21:02:20 Starting - Preparing the instances for training
2025-07-06 21:02:20 Downloading - Downloading the training image
2025-07-06 21:02:20 Training - Training image download completed. Training in progress.
2025-07-06 21:02:20 Uploading - Uploading generated training model
2025-07-06 21:02:20 Completed - Training job completed
Best model artifact at: s3://sagemaker-us-east-1-291480921130/XGBoost-Regressor/output/sagemaker-xgboost-250706-1459-001-df91f3f6/output/model.tar.gz
"""