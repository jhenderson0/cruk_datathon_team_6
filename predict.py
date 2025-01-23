import itertools

import numpy.typing as npt
import pandas as pd
import torch

from src.dumpy import Dumpy, DumpyJoint


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


def get_tcrs(path: str, alphabet: pd.DataFrame) -> torch.Tensor:
    """
    Get cdr3 from Decombinator output, strip, and translate to binary.
    """
    df = pd.read_csv(path, compression="gzip", header=0, sep="\t")
    cdr3s = df["junction_aa"].dropna().tolist()
    cdr3s = [cdr3 for cdr3 in cdr3s if len(cdr3) == 15]
    cdr3s = [mer[4:14] for mer in cdr3s]
    translate_dict = {
        aa: str(binary["Letter"]) for aa, binary in alphabet.to_dict().items()
    }
    translation_table = str.maketrans(translate_dict)
    cdr3s = [mer.translate(translation_table) for mer in cdr3s]
    cdr3s = [list(string) for string in cdr3s]
    cdr3s = [[float(i) for i in mer] for mer in cdr3s]
    cdr3s = torch.stack([torch.tensor(mer, dtype=torch.float32) for mer in cdr3s])
    return cdr3s


if __name__ == "__main__":
    model = Dumpy(5, 10)
    model.load_state_dict(
        torch.load("model_saves/dumpy_of_choice.pth", weights_only=True)
    )
    alphabet = pd.read_csv("model_saves/alphabet.csv", index_col=0)
    num_ones_in_alphabet = int(alphabet.loc["Letter"].sum())
    epitope_priors = compute_epitope_priors(5, num_ones_in_alphabet)
    joint_model = DumpyJoint(model, epitope_priors)

    tcrs = get_tcrs("./ignore/dcr_LTX_0001_N_beta.tsv.gz", alphabet)

    all_possible_epitopes = torch.stack(
        [
            torch.tensor(epitope, dtype=torch.float32)
            for epitope in epitope_priors.keys()
        ]
    )

    mini_tcr = tcrs[:5]
    mini_epitope = all_possible_epitopes[:5]

    predict = joint_model.joint(mini_tcr, mini_epitope)
    print(predict)

    # Get vs all epitopes and normalise one of interest by whole row
