import data_loader
import etl
import train

def main():
    data_loader.main()
    etl.main()
    train.main()


if __name__ == "__main__":
    main()