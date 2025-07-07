from src import data_loader, etl, train, sagemaker_prep, sagemaker_train, download_model

def main():
    data_loader.main()
    etl.main()
    train.main()
    sagemaker_prep.main()
    sagemaker_train.main()
    download_model.main()

if __name__ == "__main__":
    main()