import itertools
from pandas import DataFrame
import torch
from torch import Tensor
from torch.nn import Module, Linear
from torch import special


class Dumpy(Module):
    """
    A multivariate and multi-output logistic regression model.
    Used to model the conditional TCR distribution given peptide (of a class I human pMHC).
    """

    def __init__(self, in_features: int, out_features: int):
        super(Dumpy, self).__init__()
        self.linear = Linear(in_features=in_features, out_features=out_features, bias=True)
    
    def forward(self, x):
        x = self.linear(x)
        return special.expit(x)

    def forward_logit(self, x):
        return self.linear(x)


class DumpyJoint:
    def __init__(self, dumpy_model: Dumpy, epitope_priors: dict[tuple, float]):
        self.dumpy_model = dumpy_model
        self.tcr_dimensionality = dumpy_model.linear.out_features
        self.epitope_dimensionality = dumpy_model.linear.in_features
        self.epitope_priors = epitope_priors

    @torch.no_grad()
    def joint(self, tcrs: Tensor, epitopes: Tensor) -> Tensor:
        """
        Parameters
        ----------
        tcrs : Tensor
            A tensor of shape (n_tcrs, n_tcr_features) representing TCRs.

        epitopes : Tensor
            A tensor of shape (n_epitopes, n_epitope_features) representing epitopes.

        Returns
        -------
        Tensor
            A matrix of joint probabilities of TCRs and epitopes.
            The shape of the output matrix is (n_tcrs, n_epitopes).
            That is, this function computes the probability of all pairs between input TCRs and epitopes.
        """
        tcr_distributions = self.dumpy_model.forward(epitopes)
        conditional_tcr_probs = self._get_likelihood(tcrs, tcr_distributions)
        marginal_epitope_probs = self._get_epitope_marginals(epitopes)
        return marginal_epitope_probs.unsqueeze(0) * conditional_tcr_probs

    def _get_likelihood(self, target: Tensor, distribution: Tensor) -> Tensor:
        """
        Parameters
        ----------
        target : Tensor
            A tensor of shape (n_instances, n_features) containing the instnance vectors (e.g. TCRs).

        distribution : Tensor
            A tensor of shape (n_generators, n_features) containing vectors representing generative distributions (e.g. TCR distributions).
        
        Returns
        -------
        Tensor
            A tensor of shape (n_instances, n_generators) containing the likelihood of each instance given each of the generators.
        """
        distribution = distribution.unsqueeze(0)
        target = target.unsqueeze(1)
        return torch.prod(target * distribution + (1 - target) * (1 - distribution), dim=-1)

    def _get_epitope_marginals(self, epitopes: Tensor) -> Tensor:
        epitopes_as_tuples = [tuple(epitope.to(int).tolist()) for epitope in epitopes]
        marginals = [self.epitope_priors[epitope] for epitope in epitopes_as_tuples]
        return torch.tensor(marginals)


def compute_epitope_priors_from_alphabet(alphabet_df: DataFrame) -> dict:
    """
    Compute the prior probability of each epitope given the alphabet table.
    Note that the zeroth colum of the alphabet CSV should be read in as the index column or this won't work.
    """
    num_ones_in_alphabet = int(alphabet_df.loc["Letter"].sum())
    prob_one = num_ones_in_alphabet / 20
    prob_zero = 1 - prob_one

    possible_epitopes = [tuple(prod) for prod in itertools.product([0,1], repeat=5)]
    
    def compute_prob(epitope: tuple) -> float:
        return prob_one ** epitope.count(1) * prob_zero ** epitope.count(0)
    
    return {epitope: compute_prob(epitope) for epitope in possible_epitopes}