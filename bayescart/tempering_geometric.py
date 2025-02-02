"""Bayesian CART (BCART) models solved using geometric parallel tempering,
the whole posterior distribution is flattened uniformly.

This module implements the geometric parallel tempering algorithm for addressing 
the multimodality of BCART posterior space.
"""

import numpy as np
from .parallel_tempering_base import BCARTPT
from .node import Node
from .tree import Tree
from functools import wraps

class BCARTGeom(BCARTPT):
    """
    Geometric parallel tempering implementation of Bayesian CART.

    In this variant, the posterior of each chain is raised to a power controlled by the temperature. This yields a flattening effect. Chains furthest from the original one (t=1) are designed to explore the space fully. However, this leads to the search over very big trees, which are computationally expensive. It also slows down the convergence of the original chain (beyond runtime, because it is difficult to undo big trees).
    """

    @wraps(BCARTPT.__init__)
    def __init__(self, *args, **kwargs):
        self.tempering = 'geom'
        super().__init__(*args, **kwargs)

    def calc_grow_prune_mh_data(self, new_tree: Tree, node_to_split: Node, 
                          tot_parents_with_two_leaves: int, b: int):
        n_avail_vars, n_splits = node_to_split.calc_avail_split_and_vars()
        super().calc_grow_prune_mh_data(new_tree, node_to_split, tot_parents_with_two_leaves, b)
        self.mh_move_data['n_avail_vars'] = n_avail_vars
        self.mh_move_data['n_splits'] = n_splits

    def calc_change_swap_mh_data(self, transition_p: float):
        self.mh_move_data = {'transition_p': transition_p}

    def trans_prob_log(self, move: str, new_tree: Tree, old_tree: Tree) -> float:
        if move in ['change', 'swap']:
            trans_ratio = self.mh_move_data['transition_p']
            str_ratio = 1 / trans_ratio
            res = trans_ratio * str_ratio ** new_tree.temperature
            return np.log(res)
        
        alpha, beta = self.alpha, self.beta
        b = self.mh_move_data['b']
        depth = self.mh_move_data['depth']
        n_int_with_2_child = self.mh_move_data['n_int_with_2_child']
        n_avail_vars = self.mh_move_data['n_avail_vars']
        n_splits = self.mh_move_data['n_splits']

        trans_ratio = b * n_avail_vars * n_splits / n_int_with_2_child
        str_ratio = np.log(alpha) + 2* np.log(1-alpha/(2+depth)**beta) - (np.log((1+depth)**beta - alpha) + np.log(n_avail_vars) + np.log(n_splits))
        res = np.log(trans_ratio) + new_tree.temperature * str_ratio

        if move == 'grow':
            return np.log(self.move_prob[1]/self.move_prob[0])+res
        else:
            return np.log(self.move_prob[0]/self.move_prob[1])-res
        
    def get_swap_acceptance_prob(self, tree_from: Tree, tree_to: Tree) -> float:
        """
        Compute the acceptance probability for a swap between chains in the geometric tempering variant.

        Parameters
        ----------
        tree_from : Tree
        tree_to : Tree

        Returns
        -------
        float
            The swap acceptance probability.
        """
        chain_from_accept = self.calc_llik(tree_to) - self.calc_llik(tree_from) + self.calc_log_tree_prob(tree_to) - self.calc_log_tree_prob(tree_from)
        chain_from_accept = chain_from_accept * tree_from.temperature

        chain_to_accept = self.calc_llik(tree_from) - self.calc_llik(tree_to) + self.calc_log_tree_prob(tree_from) - self.calc_log_tree_prob(tree_to)
        chain_to_accept = chain_to_accept * tree_to.temperature

        return np.exp(chain_from_accept + chain_to_accept)
        