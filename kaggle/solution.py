# solution.py
# Carolyne Pelletier
# Isabelle Bouchard
# 2019-10-15
# IFT-6390

import numpy as np
import pathlib


DATA_PATH = pathlib.Path("data/")

def read_data(set_="train"):
    return np.load(DATA_PATH / f"data_{set_}.pkl", allow_pickle=True)

def main():
    X_train, y_train = read_data(set_="train")
    X_test = read_data(set_="test")

if __name__ == "__main__":
    main()
