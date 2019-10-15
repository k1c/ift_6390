# decipher.py
# Isabelle Bouchard, 2019-10-18
# COMP550 - Homework 2

import argparse
import nltk
import pathlib
import string
from nltk.tag import hmm
from nltk.probability import LaplaceProbDist

_ALPHABET = string.ascii_lowercase + ". ,"

def train(cipher_train_data, plain_train_data,
          cipher_test_data, plain_test_data, laplace_smoothing=False):

    train_data = prepare_data(cipher_train_data, plain_train_data)

    trainer = hmm.HiddenMarkovModelTrainer(states=_ALPHABET, symbols=_ALPHABET)
    # Trainer uses MLE by default
    if laplace_smoothing:
        tagger = trainer.train_supervised(train_data, estimator=LaplaceProbDist)
    else:
        tagger = trainer.train_supervised(train_data)

    correct = 0
    total = 0
    # Test model
    for s_cipher, s_plain in zip(cipher_test_data, plain_test_data):
        cipher, decoded = list(zip(*tagger.tag([c for c in s_cipher])))
        decoded = "".join(decoded)
        cipher = "".join(cipher)
        print("\n")
        print(f"Cipher - {cipher}")
        print(f"Plain - {s_plain}")
        print(f"Prediction - {decoded}")

        for c_decoded, c_plain in zip(decoded, s_plain):
            if c_decoded == c_plain:
                correct += 1
        total += len(decoded)

    print(f"\n>>> Accuracy {correct/total}")


def read_data(cipher_folder, set_, format_):
    datapath = pathlib.Path(f"a2data")

    with open(datapath / f"{cipher_folder}" / f"{set_}_{format_}.txt", "r") as f:
        data = f.read()
    # Split on lines
    data = data.split("\n")
    # Removes last sentence if empty (extra "\n" at the end of the file)
    data = data if data[-1] else data[:-1]
    return data


def prepare_data(cipher_data, plain_data, n=1):
    data = []
    for s_cipher, s_plain in zip(cipher_data, plain_data):
        data.append(list(zip(s_cipher, s_plain)))
    return data


def main(cipher_folder, laplace_smoothing=False, improved_lm=False):
    # Train data
    cipher_train_data = read_data(cipher_folder, "train", "cipher")
    plain_train_data = read_data(cipher_folder, "train", "plain")
    assert len(cipher_train_data) == len(plain_train_data)
    # Test data
    cipher_test_data = read_data(cipher_folder, "test", "cipher")
    plain_test_data = read_data(cipher_folder, "test", "plain")
    assert len(cipher_test_data) == len(plain_test_data)

    train(cipher_train_data,
          plain_train_data,
          cipher_test_data,
          plain_test_data,
          laplace_smoothing=laplace_smoothing)


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "cipher_folder",
        type=str,
        help="Cipher folder name: 'cipher1', 'cipher2' or 'cipher3'",
    )
    parser.add_argument(
        "--laplace",
        action="store_true",
        help="Whether or not to apply Laplace smoothing",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        help="Whether or not to improve the language modelling",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(cipher_folder=args.cipher_folder, laplace_smoothing=args.laplace, improved_lm=args.lm)
