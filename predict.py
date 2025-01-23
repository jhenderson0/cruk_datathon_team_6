from src.dumpy import Dumpy, DumpyJoint
import torch

def compute_epitope_priors(epi_length: int, num_ones_in_alphabet: int) -> dict:
    """
    Compute the prior probability of each epitope of length `epi_length` given the alphabet.
    """
    prob_one = num_ones_in_alphabet / 20
    prob_zero = 1 - prob_one

    possible_epitopes = [tuple(prod) for prod in itertools.product([0,1], repeat=epi_length)]
    
    def compute_prob(epitope: tuple) -> float:
        return prob_one ** epitope.count(1) * prob_zero ** epitope.count(0)
    
    return {epitope: compute_prob(epitope) for epitope in possible_epitopes}

if __name__ == "__main__":
    model = Dumpy(5, 10)
    model.load_state_dict(torch.load("model_saves/model_saves/dumpy_of_choice.pth", weights_only=True))
    alphabet = pd.read_csv("model_saves/alphabet.csv", index_col=0)
    num_ones_in_alphabet = int(alphabet.loc["Letter"].sum())
    epitope_priors = compute_epitope_priors(5, num_ones_in_alphabet)
    joint_model = DumpyJoint(model, epitope_priors)
    tcrs = torch
