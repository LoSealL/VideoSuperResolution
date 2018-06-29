from VSR.DataLoader.Dataset import *


def main():
    try:
        DATASETS = load_datasets('./Data/datasets.json')
    except FileNotFoundError:
        DATASETS = load_datasets('../Data/datasets.json')
    print(DATASETS['91-IMAGE'].train)


if __name__ == '__main__':
    main()
