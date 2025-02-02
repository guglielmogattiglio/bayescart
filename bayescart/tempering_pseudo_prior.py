"""Bayesian CART (BCART) models solved using tempering with a pseudo-prior.
Only the likelihood is flattened, thereby preserving prior information on the importnance
of different tree sizes. The prior is left unchanged as it should be used to inform the 
model with previous information. Instead, a pseudo-prior is used instrumentally 
to guide the search over small trees. 

We take as pseudo prior the same CGM98 prior to benefit from substantial cancellations 
in computing acceptance rates.

This module implements the tempering with a pseudo-prior algorithm for addressing 
the multimodality of BCART posterior space.
"""

import numpy as np
from .parallel_tempering_base import BCARTPT
from .node import Node
from .tree import Tree
from typing import Iterable
from functools import wraps

class BCARTPseudoPrior(BCARTPT):
    """
    Bayesian CART with a pseudo-prior used to bias the search towards smaller trees.

    The pseudo-prior is used instrumentally in computing the acceptance probabilities.
    
    Parameters
    ----------
    pprior_alpha : float, optional
        Pseudo-prior alpha parameter (default 0.95).
    pprior_beta : float, optional
        Pseudo-prior beta parameter (default 1).
    """
    @wraps(BCARTPT.__init__)
    def __init__(self, *args, pprior_alpha: float = 0.95, pprior_beta: float = 1, **kwargs):
        self.tempering = 'pseudoprior'
        self.pprior_alpha = pprior_alpha
        self.pprior_beta = pprior_beta
        super().__init__(*args, **kwargs)

    def get_acceptance_prob(self, new_tree: Tree, move: str) -> float:
        """
        Compute the acceptance probability for a move using the pseudo-prior adjustment.

        Parameters
        ----------
        new_tree : Tree
        move : str

        Returns
        -------
        float
            The acceptance probability.
        """
        # compute log-likelihood of proposed tree
        proposal_llik = self.calc_llik(new_tree)
        current_llik = self.calc_llik(self.tree)

        if self.debug:
            assert self.tree.temperature == new_tree.temperature

        trans_prob = self.trans_prob_log_pseudoprior(move, new_tree, self.tree)
        # We need additional computation based on move type
        if move in ['grow', 'prune']:
            depth = self.mh_move_data['depth']
            add_bit_prior = np.log(self.alpha) + 2* np.log(1-self.alpha/(2+depth)**self.beta) - np.log((1+depth)**self.beta - self.alpha)
            add_bit_p_prior = np.log(self.pprior_alpha) + 2* np.log(1-self.pprior_alpha/(2+depth)**self.pprior_beta) - np.log((1+depth)**self.pprior_beta - self.pprior_alpha)
            adj = add_bit_prior - add_bit_p_prior
            if move == 'prune':
                adj = -adj
        else:
            adj = 0
        a_prob = np.exp(trans_prob + new_tree.temperature * (proposal_llik - current_llik + adj))
        return a_prob

    def trans_prob_log_pseudoprior(self, move: str, new_tree: Tree, old_tree: Tree) -> float:
        """
        Compute the log transition probability for a move using the pseudo-prior.

        Parameters
        ----------
        move : str
        new_tree : Tree
        old_tree : Tree

        Returns
        -------
        float
            The log transition probability.
        """
        if move in ['change', 'swap']:
            return 0
        
        # Process Grow and Prune. 
        # Compute only the former as the latter is just the inverse.
        alpha, beta = self.pprior_alpha, self.pprior_beta
        b = self.mh_move_data['b']
        depth = self.mh_move_data['depth']
        n_int_with_2_child = self.mh_move_data['n_int_with_2_child']

        res = np.log(b) + np.log(alpha) + 2* np.log(1-alpha/(2+depth)**beta) - np.log((1+depth)**beta - alpha) - np.log(n_int_with_2_child)

        if move == 'grow':
            return np.log(self.move_prob[1]/self.move_prob[0])+res
        else:
            return np.log(self.move_prob[0]/self.move_prob[1])-res
        
    def get_swap_acceptance_prob(self, tree_from: Tree, tree_to: Tree) -> float:
        """
        Compute the swap acceptance probability for the pseudo-prior variant.

        Parameters
        ----------
        tree_from : Tree
        tree_to : Tree

        Returns
        -------
        float
            The swap acceptance probability.
        """
        
        temp_diff = tree_from.temperature - tree_to.temperature
        llik_ratio = temp_diff * (self.calc_llik(tree_to) - self.calc_llik(tree_from))
        priors_ratio = temp_diff * (self.calc_log_p_split(tree_to, self.alpha, self.beta) - self.calc_log_p_split(tree_from, self.alpha, self.beta) + self.calc_log_p_split(tree_from, self.pprior_alpha, self.pprior_beta) - self.calc_log_p_split(tree_to, self.pprior_alpha, self.pprior_beta))

        return np.exp(llik_ratio + priors_ratio)
    
    def calc_log_p_split(self, tree: Tree, alpha: float, beta: float) -> float:
        """
        Compute the log prior probability of the tree structure (ignoring the specific split values).

        Parameters
        ----------
        tree : Tree
        alpha : float
        beta : float

        Returns
        -------
        float
            The computed log prior probability of the tree structure.
        """

        if tree.is_stump():
            return np.log(1-(alpha*1**(-beta)))

        node_itr: Iterable[Node] = tree.all_nodes_itr()
        res = 0
        for node in node_itr:
            if node.is_leaf():
                res += np.log(1-(alpha*(node.depth+1)**(-beta)))
            else:
                res += np.log(alpha) - beta * np.log(1+node.depth)
        return res
    
    def run(self):
        res = super().run()
        res['setup'].update({'pprior_alpha': self.pprior_alpha, 'pprior_beta': self.pprior_beta})
        return res
                
