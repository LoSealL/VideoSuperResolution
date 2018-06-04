from VSR.DataLoader.Dataset import *


def main():
    datasets = load_datasets('../Data/datasets.json')
    print(datasets['91-IMAGE'].train)


if __name__ == '__main__':
    main()
