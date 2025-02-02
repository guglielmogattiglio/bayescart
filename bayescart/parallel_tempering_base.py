"""Bayesian CART (BCART) models solved using parallel tempering.

This module implements the parallel tempering algorithm for addressing 
the multimodality of BCART posterior space. This is an abstract base class.
Subclasses will implement the specific parallel tempering logic, such as
how to apply flattening of the posterior distribution.
"""

import numpy as np
from typing import Sequence
import numpy.typing as npt
from .bcart import BCARTClassic
from .tree import Tree
from functools import wraps

class BCARTPT(BCARTClassic):
    """
    Base class for Parallel Tempering variants of BCART.

    Supports staggered activation of chains through steps_per_chain.
    This can help warmer chains to converge before adding more chains 
    closer to the cold one (target chain).
    
    Parameters
    ----------
    temps : Sequence[float] or NDArrayFloat
        List of temperatures for each chain. One of the ends must be 1.
    steps_per_chain : Sequence[int] or NDArray[int] or None, optional
        Number of iterations to run before unlocking additional chains.
    swap_type : str, optional
        Swap type: 'stochastic' or deterministic (default 'stochastic').
    """
    tempering = ''

    @wraps(BCARTClassic.__init__)
    def __init__(self, *args, temps: Sequence[float] | npt.NDArray[np.float64], steps_per_chain: Sequence[int] | npt.NDArray[np.int_] | None = None, swap_type: str = 'stochastic', iters: int = 1250, **kwargs):
        '''
        temps: list of temperatures for each chain. 

        steps_per_chain: let lower chains converge first without working on upper ones, 
        once they converge then move up with the others chains. 
        This is always done in pairs to allow for meaningful swaps. It should contains 
        the number of iters you want each pair to run before unlocking the next. 
        For example: 8 chains, tot iters = 100, first 2 10 iters, 
        then 25 iters with next two, then 30 with next two and finally 35 with 
        the upper ones. steps_per_chain = c(10,25,30,35)'''
        n_chains = len(temps)

        if temps[0] != 1:
            if temps[-1] == 1:
                temps = temps[::-1]
            else:
                raise ValueError('Either the first or last temperature must be 1')

        if n_chains % 2 != 0:
            raise ValueError('You must use an even number of chains for tempering.')
        
        if steps_per_chain is None:
            steps_per_chain = np.concatenate([np.zeros(n_chains//2-1, dtype=int), np.array([iters])])

        if sum(steps_per_chain) != iters:
            raise ValueError('Invalid input type for steps_per_chain, sum != iters')
        
        if len(steps_per_chain) != n_chains//2:
            raise ValueError('Invalid input type for steps_per_chain, length != n_chain/2')
        
        steps_per_chain = np.concatenate([np.array([0]), np.cumsum(steps_per_chain)[:-1], np.array([iters+1])])
        
        # numer of active chains
        n_active_chains = 2 * sum(steps_per_chain == 0)

        # which iteration whould we add more chains?
        next_chain_release = steps_per_chain[int(n_active_chains/2)]

        self.temps = temps
        self.n_chains = n_chains
        self.steps_per_chain = steps_per_chain
        self.swap_type = swap_type
        self.next_chain_release = next_chain_release
        self.n_active_chains = n_active_chains

        super().__init__(*args, iters=iters, **kwargs)

    def _init(self):
        super()._init()

        # This tracks exchanges across chains
        self.PT_swap_attempts = np.zeros((self.n_chains,), dtype=int)
        self.PT_swap_accepts = np.zeros((self.n_chains,), dtype=int)
        self.PT_swap_prob_over_time = {}
        self.PT_term_reg = np.zeros((self.n_chains,1000), dtype=int)
        self.PT_term_reg_counter = 0

        # this tracks all trees
        self.trees = [self.tree.copy(light=True) for _ in range(self.n_chains)]

        # this tracks the tree in the original chain - no need to change previous information storing code
        self.tree = self.trees[0]

    def _update_once(self, move:str) -> None:
        """
        Perform one parallel tempering update step across all active chains.
        
        This includes updating each chain and attempting swaps between adjacent chains.
        """

        n_active_chains = self.n_active_chains
        current_iter = self.current_iter

        # Unlock new chains if possible
        if current_iter == self.next_chain_release:
            # how many chains are we going to unlock?
            chain_increment = 2 * sum(self.steps_per_chain == self.next_chain_release)
            for k in range(n_active_chains, chain_increment + n_active_chains):
                if self.debug:
                    assert self.trees[k].is_stump()
                # Copy the same tree as in the last chain - no reason to start from scratch
                self.trees[k] = self.trees[n_active_chains-1]
            n_active_chains += chain_increment
            self.next_chain_release = self.steps_per_chain[int(n_active_chains/2)]
            self.n_active_chains = n_active_chains


        # Update the tree in each available chain
        for chain_idx in range(n_active_chains):
            self.current_chain = chain_idx
            self.tree = self.trees[chain_idx]
            self.tree.temperature = self.temps[chain_idx]
            # sample a move
            move = self.rng.choice(['grow', 'prune', 'change', 'swap'], p=self.move_prob)

            # Force grow for the first few iterations
            if self.current_iter < 1:
                move = 'grow'


            # attempt the update. Note: the acceptance probability is overloaded.
            super()._update_once(move)
            # replace updated tree in the chain
            self.trees[chain_idx] = self.tree

        
        # Attemp to swap trees across chains. Do not do this during burnin
        # actually we want this to happen since the beginning!
        if True or current_iter >= self.burnin:
            if self.swap_type == 'stochastic':
                # Pick at random whether to swap even (E) or odd (O) chains. E-O swap
                condition = self.rng.random() <= 0.5
            else: # deterministic
                # Alternate in swapping E and O deterministically
                condition = current_iter % 2 == 0
        
            idx_swap_from = np.arange(0, n_active_chains, 2)               
            # Edit the default E swap to O swap
            if condition:
                idx_swap_from = idx_swap_from + 1
            idx_swap_to = idx_swap_from + 1
            idx_swap_to = idx_swap_to % n_active_chains

            # Iterate over the swaps
            for chain_from, chain_to in zip(idx_swap_from, idx_swap_to):
                if self.debug:
                    assert chain_from != chain_to
                self.PT_swap_attempts[chain_from] += 1

                # Compute acceptance probability of swap
                a_prob = self.get_swap_acceptance_prob(self.trees[chain_to], self.trees[chain_from])
                a_prob = min(1, a_prob)
                # Swap trees if accepted
                if self.rng.random() <= a_prob:
                    self.trees[chain_from], self.trees[chain_to] = self.trees[chain_to], self.trees[chain_from]
                    self.PT_swap_accepts[chain_from] += 1

        # Store results
        self.PT_term_reg[:,self.PT_term_reg_counter] = np.array([self.trees[k].get_n_leaves() for k in range(self.n_chains)])
        self.PT_term_reg_counter += 1
        if self.PT_term_reg_counter == 999:
            self.PT_term_reg_counter = 0
        if current_iter % 1000 == 0 and current_iter > 0:
            self.PT_swap_prob_over_time[current_iter] = self.PT_swap_accepts/self.PT_swap_attempts

        # Set as "main" tree the one in the original chain
        self.tree = self.trees[0]

    def run(self):
        """
        Run the Parallel Tempering MCMC sampler.

        Returns
        -------
        dict
            A dictionary containing MCMC results and tempering statistics.
        """
        res = super().run()
        res['PT_swap_stats'] = {'accepts': self.PT_swap_accepts, 'attempts': self.PT_swap_attempts, 'PT_swap_final_prob': np.round(self.PT_swap_accepts / self.PT_swap_attempts, 2), 'PT_swap_prob_over_time': self.PT_swap_prob_over_time, 'PT_term_reg': self.PT_term_reg}
        res['setup'].update({'temps': self.temps, 'steps_per_chain': self.steps_per_chain, 'swap_type': self.swap_type})
        return res

    def get_acceptance_prob(self, new_tree: Tree, move: str) -> float:
        """
        Compute the acceptance probability for a move in the parallel tempering framework.

        Parameters
        ----------
        new_tree : Tree
            The proposed tree.
        move : str
            The move type.

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
        
        trans_prob = self.trans_prob_log(move, new_tree, self.tree)
        a_prob = np.exp(trans_prob + new_tree.temperature * (proposal_llik - current_llik))
        return a_prob
    
    def get_swap_acceptance_prob(self, tree_from: Tree, tree_to: Tree) -> float:
        """
        Compute the swap acceptance probability between two chains.

        Parameters
        ----------
        tree_from : Tree
            The tree from one chain.
        tree_to : Tree
            The tree from the adjacent chain.

        Returns
        -------
        float
            The swap acceptance probability.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        raise NotImplementedError()

