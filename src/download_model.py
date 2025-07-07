import boto3
from pathlib import Path
import tarfile

def download_model(bucket: str, key: str, output_path: Path):
    s3 = boto3.client("s3")
    local_tar = output_path / "model.tar.gz"

    s3.download_file(Bucket=bucket, Key=key, Filename=str(local_tar))

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=output_path)

    print("Downloaded at:", output_path)

def main():
    bucket = "sagemaker-us-east-1-291480921130"
    key = "XGBoost-Regressor/output/sagemaker-xgboost-250706-1459-001-df91f3f6/output/model.tar.gz"
    base = Path().resolve().parent
    file_path = base / "model"
    download_model(bucket, key, file_path)

if __name__ == "__main__":
    main()
