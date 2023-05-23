#!/usr/bin/env python3
"""
This is a setup file that downloads necessary data.
"""
from src.data_loading import Loader10x, preprocess_matrices

def main():
    yes = input("The download may take up to 30 minutes depending on your" +
                "downloading speed. Type 'Y' to proceed: ")
    if yes == 'Y':
        print("The download has started.")
        loader = Loader10x('data/raw/')
        loader.download_dataset()
        print("All files are downloaded.")
        loader.preprocess_matrices('data/')


if __name__ == '__main__':
    main()
