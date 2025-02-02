"""Bayesian CART (BCART) models solved using geometric-likelihood parallel tempering.
Only the likelihood is flattened, thereby preserving prior information on the importnance
of different tree sizes. The prior is used instrumentally to guide the search over small trees. 

Disentangling the likelihood from the prior leads to sensible analytical simplifications and performacne gains (also in terms of convergence). However, this approach loses the ability
to leverage prior inforamtion, as it is now used to constrain smaller trees in heated chains.

This module implements the geometric-likelihood parallel tempering algorithm for addressing 
the multimodality of BCART posterior space.
"""

import numpy as np
from .parallel_tempering_base import BCARTPT
from .tree import Tree
from functools import wraps

class BCARTGeomLik(BCARTPT):
    """
    Geometric-Likelihood parallel tempering implementation.

    In this variant, only the likelihood is flattened, while the prior remains unchanged.
    """

    @wraps(BCARTPT.__init__)
    def __init__(self, *args, **kwargs):
        self.tempering = 'geomlik'
        super().__init__(*args, **kwargs)

    def get_swap_acceptance_prob(self, tree_from: Tree, tree_to: Tree) -> float:
        """
        Compute the swap acceptance probability for the geometric-likelihood variant.

        Parameters
        ----------
        tree_from : Tree
        tree_to : Tree

        Returns
        -------
        float
            The swap acceptance probability.
        """
        a_prob = (self.calc_llik(tree_to) - self.calc_llik(tree_from)) * tree_from.temperature + (self.calc_llik(tree_from) - self.calc_llik(tree_to)) * tree_to.temperature

        return np.exp(a_prob)
