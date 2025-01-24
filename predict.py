import argparse
import itertools

import numpy.typing as npt
import pandas as pd
import torch

from src.dumpy import Dumpy, DumpyJoint

PMHC_LEN = 5
TCR_LEN = 10
EPITOPE = "RLMAPVGSV"


def compute_epitope_priors(epi_length: int, num_ones_in_alphabet: int) -> dict:
    """
    Compute the prior probability of each epitope of length `epi_length` given the alphabet.
    """
    prob_one = num_ones_in_alphabet / 20
    prob_zero = 1 - prob_one

    possible_epitopes = [
        tuple(prod) for prod in itertools.product([0, 1], repeat=epi_length)
    ]

    def compute_prob(epitope: tuple) -> float:
        return prob_one ** epitope.count(1) * prob_zero ** epitope.count(0)

    return {epitope: compute_prob(epitope) for epitope in possible_epitopes}


def get_translate_table(alphabet: pd.DataFrame) -> dict[int, str]:
    """
    Returns a translation table based on alphabet provided
    """
    translate_dict = {
        aa: str(binary["Letter"]) for aa, binary in alphabet.to_dict().items()
    }
    return str.maketrans(translate_dict)


def get_tcrs(path: str, alphabet: pd.DataFrame) -> torch.Tensor:
    """
    Get cdr3 from Decombinator output, strip, and translate to binary.
    """
    df = pd.read_csv(path, compression="gzip", header=0, sep="\t")
    cdr3s = df["junction_aa"].dropna().tolist()
    cdr3s = [cdr3 for cdr3 in cdr3s if len(cdr3) == 15]
    cdr3s = [mer[4:14] for mer in cdr3s]
    table = get_translate_table(alphabet)
    cdr3s = [mer.translate(table) for mer in cdr3s]
    cdr3s = [list(string) for string in cdr3s]
    cdr3s = [[float(i) for i in mer] for mer in cdr3s]
    cdr3s = torch.stack([torch.tensor(mer, dtype=torch.float32) for mer in cdr3s])
    return cdr3s


def translate_epitope(seq: str, alphabet: pd.DataFrame) -> torch.Tensor:
    """
    Trim and translate epitope to binary
    """
    seq = seq[2:7]
    table = get_translate_table(alphabet)
    seq = seq.translate(table)
    split = list(seq)
    floats = [float(i) for i in split]
    return torch.tensor(floats, dtype=torch.float32)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dumpy prediction tool")
    parser.add_argument("--in", action="store", dest="path")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = Dumpy(PMHC_LEN, TCR_LEN)
    model.load_state_dict(
        torch.load("model_saves/dumpy_of_choice.pth", weights_only=True)
    )
    alphabet = pd.read_csv("model_saves/alphabet.csv", index_col=0)
    num_ones_in_alphabet = int(alphabet.loc["Letter"].sum())
    epitope_priors = compute_epitope_priors(5, num_ones_in_alphabet)
    joint_model = DumpyJoint(model, epitope_priors)

    tcrs = get_tcrs(args.path, alphabet)

    interest_epitope = translate_epitope(EPITOPE, alphabet)

    all_possible_epitopes = torch.stack(
        [
            torch.tensor(epitope, dtype=torch.float32)
            for epitope in epitope_priors.keys()
        ]
    )

    interest_predict = joint_model.joint(tcrs, interest_epitope.unsqueeze(0))
    all_predict = joint_model.joint(tcrs, all_possible_epitopes)

    norm = interest_predict / torch.sum(all_predict)

    savename = args.path.split("/")[-1].split(".")[0]
    torch.save(norm, f"./ignore/{savename}_interest_probability.pt")
    torch.save(all_predict, f"./ignore/{savename}_all_probability.pt")
