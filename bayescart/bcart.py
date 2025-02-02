"""Bayesian CART (BCART) models.

This module implements the BCART class and several subclasses to perform
Bayesian inference on classification and regression trees. 
Subclasses address performance improvements, it is recommended to use these.
"""

import numpy as np
import scipy
from scipy.special import gammaln
import time
import pandas as pd
from typing import Sequence, Self, Any
import numpy.typing as npt
import math
from tqdm import tqdm
import humanize
from .node import Node
from .node_data import NodeData, NodeDataRegression, NodeDataClassification, NodeDataRegressionFast, NodeDataClassificationFast
from .tree import Tree, TreeFast
from .exceptions import InvalidTreeError, AbstractMethodError
from .utils import my_choice, invgamma_rvs, invgamma_logpdf, norm_logpdf, dirichlet_logpdf
from .mytyping import NDArrayFloat, NDArrayInt, T
from functools import wraps

class BCART():
    """
    Base class for Bayesian CART (Classification and Regression Trees).

    This class provides methods for initializing the tree, running the Markov chain Monte Carlo (MCMC)
    sampling, computing likelihoods and priors, resampling parameters, and updating tree moves (grow, prune, change, swap).

    Specific MCMC-methods are abstract and should be implemented in derived classes.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Response data.
    alpha : float, optional
        Hyperparameter for the tree prior (default 0.95).
    beta : float, optional
        Hyperparameter controlling the split probability decay (default 0.5).
    mu_0 : float or None, optional
        Initial value for the regression mean (default None, will be sampled).
    sigma_0 : float or None, optional
        Initial value for the regression standard deviation (default None, will be sampled).
    nu : float, optional
        Hyperparameter for the inverse gamma prior (default 3).
    lambd : float, optional
        Scale hyperparameter for the inverse gamma prior (default 0.1).
    mu_bar : float, optional
        Prior mean for regression (default 0).
    a : float, optional
        Hyperparameter (default 0.3).
    node_min_size : int, optional
        Minimum observations per node (default 5).
    alpha_dirichlet : NDArrayLike, optional
        Parameter(s) for the Dirichlet prior in classification (default 1.0).
    iters : int, optional
        Total number of MCMC iterations (default 1250).
    burnin : int, optional
        Number of burn-in iterations (default 250).
    thinning : int, optional
        Thinning factor (default 1).
    store_tree_spacing : int, optional
        Spacing for storing tree copies (default 10).
    max_stored_trees : int, optional
        Maximum number of stored trees (default 2000). Caps the total memory consumption.
    move_prob : Sequence[float], optional
        Probabilities for the moves (grow, prune, change, swap) (default [0.25,0.25,0.25,0.25]).
    verbose : str, optional
        Verbosity level.
    seed : int or np.random.Generator, optional
        Random seed or generator (default 45).
    debug : bool, optional
        If True, enables debugging assertions (default False).
    light : bool, optional
        If True, uses lighter copies for speed (default True).

    Attributes
    ----------
    is_classification : bool
        True if running a classification task.
    K : int
        Number of classes (if classification).
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, 
                 alpha: float = 0.95, beta: float = 0.5,
                 mu_0: float|None = None, sigma_0: float|None = None,
                 nu: float = 3, lambd: float = 0.1, mu_bar: float = 0, a: float = 0.3,
                 node_min_size: int = 5, alpha_dirichlet: npt.ArrayLike = 1.0,
                 iters: int = 1250, burnin: int = 250, thinning: int = 1,
                 store_tree_spacing: int = 10, max_stored_trees: int = 2000, move_prob: Sequence[float] = [0.25, 0.25, 0.25, 0.25],
                 verbose: str = '', seed: int|np.random.Generator = 45, debug: bool = False, light: bool = True):
        """
        Initialize the BCART object with model and MCMC parameters.
        """
        
        # store_tree_spacing: how often do you want to store trees after burnin?
        # max_stored_trees: do a first-in first-out storage of trees. This caps the total memory consumption.
        
        # check input types
        # TODO

        self.nu = float(nu)
        self.lambd = float(lambd)
        self.mu_bar = float(mu_bar)
        self.a = float(a)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.iters = iters
        self.burnin = burnin
        self.thinning = thinning
        self.node_min_size = node_min_size
        self.move_prob = np.array(move_prob)
        self.move_prob = self.move_prob/self.move_prob.sum()
        self.verbose = verbose
        self.debug = debug
        self.light = light
        self.X = X
        self.y = y
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.alpha_dirichlet = np.array(alpha_dirichlet)
        self.store_tree_spacing = store_tree_spacing
        self.max_stored_trees = max_stored_trees
        self.orig_seed = seed

        if isinstance(seed, (int, float, np.integer, np.floating)):
            self.rng = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            raise ValueError(f'Seed must be an int or a numpy random generator, {type(seed)} was given..')

        self.mh_move_data: dict = {}
        self.cache_counters = {'log_tree_prior_prob_cached': 0, 'llik_cached': 0, 'log_tree_prior_prob_tot': 0, 'llik_tot': 0, 'accepted': 0, 'proposed': 0, 'failed_error': 0, 'failed_prob': 0,'grow': 0, 'prune': 0, 'change': 0, 'swap': 0, }

        # sanity check for now, you'll want to restructure the code instead
        self.has_run = False

        # check type of learning task
        self.is_classification: bool = hasattr(y, 'cat')
        K = len(y.cat.categories) if self.is_classification else 0
        self.K = K

        self._init()

    def _init(self):
        """
        Initialize the tree and associated node data based on whether the task is classification or regression.
        """
        K, X, y = self.K, self.X, self.y
        iters, burnin, sigma_0, mu_0 = self.iters, self.burnin, self.sigma_0, self.mu_0
        alpha_dirichlet = self.alpha_dirichlet
        
        # Derived classes can overload the node data class
        self.node_data_reg = getattr(self, 'node_data_reg', NodeDataRegression)
        self.node_data_class = getattr(self, 'node_data_class', NodeDataClassification)
        self.tree_class = getattr(self, 'tree_class', Tree)

        if self.is_classification:
            if len(self.verbose) > 0:
                print('Running classification trees')
            if alpha_dirichlet.size == 1:
                alpha_dirichlet = np.repeat(alpha_dirichlet, K)
            if len(alpha_dirichlet) != K:
                raise ValueError('Length of alpha_dirichlet must be equal to the number of classes')
            self.alpha_dirichlet: NDArrayFloat = alpha_dirichlet
            # parameters initialization
            p_0 = self.sample_prior_p()
            
            
            root_node_data: NodeData = self.node_data_class(X=X, y=y, p=p_0, rng=self.rng, debug=self.debug, node_min_size=self.node_min_size)

        else:
            if len(self.verbose) > 0:
                print('Running regression trees')

            #parameters initialization
            if sigma_0 is None:
                sigma_0 = self.sample_prior_sigma()
            if mu_0 is None:
                mu_0 = self.sample_prior_mu(sigma_0)

            root_node_data: NodeData = self.node_data_reg(X=X, y=y, mu=mu_0, sigma=sigma_0, rng=self.rng, debug=self.debug, node_min_size=self.node_min_size)

        if iters < burnin:
            iters = iters + burnin
        self.iters = iters

        # Initialize the tree
        self.tree: Tree = self.tree_class(root_node_data=root_node_data, rng=self.rng, node_min_size=self.node_min_size, debug=self.debug)

        
    def run(self):
        """
        Run the MCMC algorithm for Bayesian CART.

        Returns
        -------
        dict
            A dictionary containing stored trees, integrated likelihoods, terminal region counts,
            cache counters, setup parameters, and data information.
        """
        self.current_iter = 0
        start_time = time.time()
        out = self._run()
        end_time = time.time()

        tot_mh_steps = self.iters
        elap_time = end_time - start_time
        elap_time_human = humanize.precisedelta(int(elap_time))
        if self.verbose:
            print(f'Elapsed time: {elap_time_human}, Tot iters: {tot_mh_steps}, Iters/min: {int(tot_mh_steps/elap_time*60)}/min')

        timings = {'elap_time': elap_time, 'tot_mh_steps': tot_mh_steps, 'iters/min': int(tot_mh_steps/elap_time*60), 'elap_time_human': elap_time_human}
        out.update({'timings': timings})
        setup = {'alpha': self.alpha, 'beta': self.beta, 'mu_0': self.mu_0, 'sigma_0': self.sigma_0, 'nu': self.nu, 'lambd': self.lambd, 'mu_bar': self.mu_bar, 'a': self.a, 'alpha_dirichlet': self.alpha_dirichlet, 'node_min_size': self.node_min_size, 'iters': self.iters, 'burnin': self.burnin, 'thinning': self.thinning, 'store_tree_spacing': self.store_tree_spacing, 'max_stored_trees': self.max_stored_trees, 'move_prob': self.move_prob, 'seed': self.orig_seed, 'debug': self.debug, 'light': self.light, 'verbose': self.verbose}
        data = {'X': self.X, 'y': self.y}
        out.update({'cache_counters': self.cache_counters, 'setup': setup, 'data': data})
        # remove this once code has been restructured
        # self.out = out
        return out


    
    def _run(self):
        """
        Execute the main MCMC loop.

        Returns
        -------
        dict
            A dictionary with stored trees, likelihoods, and terminal region information.
        """
        self.has_run = True

        store_size = int(np.ceil((self.iters - self.burnin)/self.thinning))
        tree_store_size_needed = int(np.ceil((self.iters - self.burnin)/self.store_tree_spacing))
        tree_store = [None for i in range(min(tree_store_size_needed, self.max_stored_trees))]
        tree_store_counter = 0
        if not self.light:
            tree_prior_store = np.empty(store_size)
            tree_prob_store = np.empty(store_size)
            posterior_store = np.empty(store_size)
        integr_llik_store = np.empty(store_size)
        tree_term_reg = np.empty(store_size) # tot terminal regions/nodes
        n_vals = self.X.shape[1]

        # store counter
        c = 0
        if len(self.verbose) > 0:
            _range = tqdm(range(self.iters))
        else:
            _range = range(self.iters)
        for i in _range:
            self.current_iter = i

            # sample a move
            move = self.rng.choice(['grow', 'prune', 'change', 'swap'], p=self.move_prob)

            # Force grow for the first few iterations
            if i < 1:
                move = 'grow'

            self._update_once(move)

            if self.debug:
                assert self.tree.is_valid()

            # store data
            if (i >= self.burnin) and ((i - self.burnin) % self.thinning == 0):
                tree_term_reg[c] = self.tree.get_n_leaves()
                integr_llik_store[c] = self.calc_llik(self.tree)

                if not self.light:
                    tree_prior_store[c] = self.calc_log_tree_prob(self.tree)
                    tree_prob_store[c] = tree_prior_store[c] + integr_llik_store[c]
                    posterior_store[c] = self.get_log_posterior_prob(self.tree)
                c += 1

            # Storing trees
            if (i >= self.burnin) and ((i - self.burnin) % self.store_tree_spacing == 0):
                tree_store[tree_store_counter] = self.tree.copy(no_data=True, light=True) # type: ignore
                tree_store_counter += 1
                if tree_store_counter >= self.max_stored_trees:
                    tree_store_counter = 0

        res = {'tree_store': tree_store, 'integr_llik_store': integr_llik_store, 'tree_term_reg': tree_term_reg}
        if not self.light:
            res.update({'tree_prior_store': tree_prior_store, 'tree_prob_store': tree_prob_store, 'posterior_store': posterior_store})

        return res
    
    def _update_once(self, move: str) -> None:
        """
        Perform one MCMC update step using the specified move.

        Parameters
        ----------
        move : str
            The move type ('grow', 'prune', 'change', or 'swap').
        """
        try:
            self.cache_counters['proposed'] += 1
            new_tree = self.update_tree(move)
            success = True
        except InvalidTreeError:
            success = False
            self.cache_counters['failed_error'] += 1
            self.cache_counters[move] += 1

        if success:
            # compute the acceptance probability
            a_prob = min(self.get_acceptance_prob(new_tree, move), 1)
            self.mh_move_data: dict = {}
            if self.rng.random() <= a_prob:
                self.tree = new_tree
                self.cache_counters['accepted'] += 1
            else:
                self.cache_counters['failed_prob'] += 1

        # Update leaf parameters whether accepted or not
        self.resample_leaf_params()




    def get_acceptance_prob(self, new_tree: Tree, move: str) -> float:
        """
        Compute the acceptance probability for a proposed tree move.

        Parameters
        ----------
        new_tree : Tree
            The proposed tree after a move.
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
        
        trans_prob = self.trans_prob_log(move, new_tree, self.tree)
        a_prob = np.exp(trans_prob + (proposal_llik - current_llik))
        return a_prob


    def calc_llik(self, tree: Tree) -> float:
        """
        Compute the integrated log-likelihood P(Y|X,T) for the tree.

        Parameters
        ----------
        tree : Tree

        Returns
        -------
        float
            The integrated log-likelihood.
        """
        if tree.llik is not None:
            llik =  tree.llik
            self.cache_counters['llik_cached'] += 1
            if self.debug:
                assert llik == self._calc_llik(tree)
        else:
            llik = self._calc_llik(tree)
            tree.llik = llik
        self.cache_counters['llik_tot'] += 1
        return llik
    
    def _calc_llik(self, tree: Tree) -> float:
        # Compute integrated log likelyhood P(Y|X,T)
        if self.is_classification:
            nik = self.get_leaves_counts(tree)
            ni = nik.sum(axis=1)
            b = nik.shape[0]
            tot_alpha = self.alpha_dirichlet.sum()

            res = b*(gammaln(tot_alpha) - np.sum(gammaln(self.alpha_dirichlet))) + np.sum(np.sum(gammaln(nik + self.alpha_dirichlet.reshape(1, -1)), axis=1) - gammaln(ni + tot_alpha))
        else:
            a, nu, lambd = self.a, self.nu, self.lambd
            ni, mui, si = self.get_leaves_means(tree)
            ti = (ni * a)/(ni + a)*(mui - self.mu_bar)**2
            res = -ni/2*np.log(np.pi) + nu/2*np.log(nu*lambd) + 0.5*np.log(a) - 0.5*np.log(ni + a) + gammaln((ni+nu)/2) - gammaln(nu/2) - (ni+nu)/2*np.log(si + ti + nu*lambd)
            res = np.sum(res)
        return res


    def get_leaves_counts(self, tree: Tree) -> NDArrayInt:
        """
        For classification, compute the counts of observations per class in each leaf.

        Parameters
        ----------
        tree : Tree

        Returns
        -------
        NDArrayInt
            Array of shape (n_leaves, n_classes).
        """

        leaves = tree.get_leaves()
        l = []
        for leaf in leaves:
            class_counts = leaf.count_values()
            l.append(class_counts)
            if self.debug:
                assert (class_counts).sum() == leaf.get_nobs()
        out = np.array(l)
        if self.debug:
            assert out.shape[0] == len(leaves)
            assert out.sum() == self.y.shape[0]
        return out


    def get_leaves_means(self, tree: Tree) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        For regression, compute for each leaf the number of observations, the mean, and un-normalized variance.

        Parameters
        ----------
        tree : Tree

        Returns
        -------
        tuple
            (n_obs per leaf, means per leaf, un-normalized variances per leaf)
        """
        leaves = tree.get_leaves()
        l = []
        for leaf in leaves:
            ni, mui, si = leaf.get_data_averages()
            l.append([ni, mui, si])
            if self.debug:
                assert ni == leaf.get_nobs()
        l = np.array(l).T
        return l[0,:], l[1,:], l[2,:]

        
        
    def trans_prob_log(self, move: str, new_tree: Tree, old_tree: Tree) -> float:
        """
        (Abstract in this base class) Compute the log transition probability for a move.

        Parameters
        ----------
        move : str
            Move type.
        new_tree : Tree
            Proposed tree.
        old_tree : Tree
            Current tree.

        Returns
        -------
        float
            The log transition probability.

        Raises
        ------
        AbstractMethodError
            Always raised; to be implemented in a subclass.
        """
        raise AbstractMethodError()
    


    def sample_prior_p(self) -> NDArrayFloat:
        """
        Sample initial class probabilities from a Dirichlet prior.

        Returns
        -------
        NDArrayFloat
            The sampled probability vector.
        """
        return self.rng.dirichlet(self.alpha_dirichlet)
    
    def sample_prior_sigma(self) -> float:
        """
        Sample an initial sigma value from the inverse gamma prior.

        Returns
        -------
        float
            The sampled sigma.
        """
        if self.debug:
            assert isinstance(self.nu, float) and isinstance(self.lambd, float)
        # a = shape
        return np.sqrt(scipy.stats.invgamma.rvs(a=self.nu/2, scale=self.nu*self.lambd/2, random_state=self.rng))
    
    def sample_prior_mu(self, sigma: float) -> float:
        """
        Sample an initial mu value from the normal prior.

        Parameters
        ----------
        sigma : float
            The sigma value used in the variance of the normal prior.

        Returns
        -------
        float
            The sampled mu.
        """
        if self.debug:
            assert isinstance(self.mu_bar, float) and isinstance(self.a, float) and isinstance(sigma, float)
        return self.rng.normal(self.mu_bar, sigma/np.sqrt(self.a), 1)[0]

    def copy(self) -> Self:
        """
        Create a copy of the BCART object.

        Returns
        -------
        Self
            A copy of the current BCART instance.
        
        Raises
        ------
        NotImplementedError
            To be implemented.
        """
        raise NotImplementedError()
    
    def deep_copy(self) -> Self:
        """
        Create a deep copy of the BCART object.

        Returns
        -------
        Self
            A deep copy.
        
        Raises
        ------
        NotImplementedError
            To be implemented.
        """
        raise NotImplementedError()

    def grow_copy(self, *args, light=False, **kwargs) -> Tree:
        """
        Pick a leaf which has > node_min_size observations. Split it using the splitting rule. Return a copied tree.

        Returns
        -------
        Tree
            The new tree after the grow move.
        """

        # copy the tree
        new_tree = self.tree.copy(light=light)

        # sanity check
        if self.debug:
            if len(self.mh_move_data) != 0:
                raise ValueError('mh_move_data should be Empty. BUG!!')
        
        # sample a leaf to split
        node_to_split = new_tree.sample_leaf(self.node_min_size)
        
        if self.debug:
            assert node_to_split.get_nobs() >= self.node_min_size

        # sample a new split for the node
        split_var, split_val = node_to_split.get_new_split()

        # Generate node parameters
        l_leaf_params, r_leaf_params = self.sample_leaf_params()

        # update the tree (apply split). This should be done inside Tree.
        new_tree.apply_split(node_to_split, split_var, split_val, l_leaf_params, r_leaf_params)

        if self.debug:
            assert self.tree.get_n_leaves()+1 == new_tree.get_n_leaves()

        # compute MH info data
        # depth of node to split
        # number of parents with two leaves on the new tree
        # number of leaves in the original tree
        self.calc_grow_prune_mh_data(new_tree=new_tree,
                               node_to_split=node_to_split,
                               tot_parents_with_two_leaves=len(new_tree.get_parents_with_two_leaves()), 
                               b = self.tree.get_n_leaves())
        return new_tree
    

    def prune_copy(self, *args, light=False, **kwargs) -> Tree:  
        """
        Pick a node with two leaves as children and prune them. Return a copied tree.

        Returns
        -------
        Tree
            The new tree after the prune move.
        """
        # copy the tree
        new_tree = self.tree.copy(light=light)

        # sanity check
        if self.debug:
            if len(self.mh_move_data) != 0:
                raise ValueError('mh_move_data should be empty. BUG!!')
        
        # no operation can be done on a stump
        if new_tree.is_stump():
            raise InvalidTreeError('Tree has only one node. Cannot prune.')
        
        # get all nodes that have two leaves as children
        cands = new_tree.get_parents_with_two_leaves()
        if len(cands) == 0:
            raise InvalidTreeError('No available nodes to prune')
        
        # choose a candidate to prune
        to_prune: Node = my_choice(self.rng, cands)
        children = new_tree.get_children(to_prune)

        if self.debug:
            assert not to_prune.is_leaf()
            assert len(children) == 2
            assert sum([c.is_leaf() for c in children ]) == 2

        # remove the children
        # Note: use remove_subtree() to get the removed part of the tree and reverse the operation
        removed = 0
        removed += new_tree.remove_node(children[0])
        removed += new_tree.remove_node(children[1])

        if self.debug:
            assert removed == 2
            assert new_tree.get_n_leaves() + 1 == self.tree.get_n_leaves()

        # compute MH info data
        # depth of node to prune
        # number of parents with two leaves on the original tree
        # number of leaves in the proposal tree
        self.calc_grow_prune_mh_data(new_tree=new_tree,
                               node_to_split=to_prune,
                               tot_parents_with_two_leaves=len(cands), 
                               b = new_tree.get_n_leaves())
        
        # Remove splitting infor from pruned node - it is now a leaf
        to_prune.reset_split_info()
        
        return new_tree
    
    def change_copy(self, *args, light=False, **kwargs) -> Tree:
        """
        Pick an internal node and change its splitting rule. Return a copied tree.

        Returns
        -------
        Tree
            The new tree after the change move.
        """
        # copy the tree
        new_tree = self.tree.copy(light=light)

        # sanity check
        if self.debug:
            if len(self.mh_move_data) != 0:
                raise ValueError('mh_move_data should be empty. BUG!!')
        
        # no operation can be done on a stump
        if new_tree.is_stump():
            raise InvalidTreeError('Tree has only one node. Cannot prune.')

        cands = new_tree.get_nonleaf_nodes()
        if self.debug:
            assert len(cands) > 0

        # choose a node to change
        to_change: Node = my_choice(self.rng, cands)

        # sample a new split for the node
        split_var, split_val = to_change.get_new_split()

        # Save the number of available splits and vars for Tree @ to_change
        _, tree_n_splits = to_change.calc_avail_split_and_vars()

        # tentatively replace the split rule
        new_tree.update_split(to_change, split_var, split_val)

        # Compute the number of available splits and vars for new_tree @ to_change
        _, new_tree_n_splits = to_change.calc_avail_split_and_vars()
        transition_p = new_tree_n_splits / tree_n_splits

        # compute MH info data
        self.calc_change_swap_mh_data(transition_p=transition_p)
    
        return new_tree


    def swap_copy(self, *args, light=False, **kwargs) -> Tree:
        """
        Pick a parent-child of internal nodes and swap their splitting rules. If the otehr child has the same rule, then swap parent with both children. Return a copied tree.

        Returns
        -------
        Tree
            The new tree after the swap move.
        """
        
        # copy the tree
        new_tree = self.tree.copy(light=light)

        # sanity check
        if self.debug:
            if len(self.mh_move_data) != 0:
                raise ValueError('mh_move_data should be empty. BUG!!')
            
        # no operation can be done on a stump
        if new_tree.is_stump():
            raise InvalidTreeError('Tree has only one node. Cannot swap.')
        
        cands = new_tree.get_nonleaf_nodes(filter_root=True)
        if len(cands) == 0:
            raise InvalidTreeError('No available nodes to swap')
        
        # first child
        c1 = my_choice(self.rng, cands)
        parent = new_tree.get_parent(c1)
        c2 = new_tree.get_sibling(c1)

        # check if c1 and c2 have the same split rule
        has_same_rule = False
        c1_s_var, c1_s_val, is_cat = c1.get_split_info()
        if not is_cat: c1_s_val = [c1_s_val]
        c2_s_var, c2_s_val, is_cat = c2.get_split_info()
        if not is_cat: c2_s_val = [c2_s_val]
        if c1_s_var == c2_s_var:
            if set(c1_s_val) == set(c2_s_val):
                has_same_rule = True

        # store MH data for tree
        tree_par_nvars, tree_par_nsplits = parent.calc_avail_split_and_vars()
        tree_c1_nvars, tree_c1_nsplits = c1.calc_avail_split_and_vars()
        if has_same_rule:
            tree_c2_nvars, tree_c2_nsplits = c2.calc_avail_split_and_vars()
        
        # swap rules
        par_s_var, par_s_val, _ = parent.get_split_info()
        if has_same_rule:
            c2.update_split_info(par_s_var, par_s_val)
        temp_s_var, temp_s_val, _ = c1.get_split_info()
        c1.update_split_info(par_s_var, par_s_val)
        parent.update_split_info(temp_s_var, temp_s_val)

        # update data splits recursively
        new_tree.update_subtree_data(parent)
    
        # Compute MH data for new_tree
        new_tree_par_nvars, new_tree_par_nsplits = parent.calc_avail_split_and_vars()
        new_tree_c1_nvars, new_tree_c1_nsplits = c1.calc_avail_split_and_vars()
        if has_same_rule:
            new_tree_c2_nvars, new_tree_c2_nsplits = c2.calc_avail_split_and_vars()
        else:
            tree_c2_nvars, tree_c2_nsplits, new_tree_c2_nvars, new_tree_c2_nsplits = 1, 1, 1, 1

        if self.debug:
            assert new_tree_par_nvars == tree_par_nvars

        numerator = - np.log(np.array([tree_par_nvars, tree_par_nsplits, tree_c1_nvars, tree_c1_nsplits, tree_c2_nvars, tree_c2_nsplits]))

        denominator = np.log(np.array([new_tree_par_nvars, new_tree_par_nsplits, new_tree_c1_nvars, new_tree_c1_nsplits, new_tree_c2_nvars, new_tree_c2_nsplits]))

        transition_p = np.exp(numerator.sum() + denominator.sum())

        if self.debug:
            assert np.isclose(transition_p, (new_tree_par_nvars * new_tree_par_nsplits * new_tree_c1_nvars * new_tree_c1_nsplits * new_tree_c2_nvars * new_tree_c2_nsplits)/(tree_par_nvars * tree_par_nsplits * tree_c1_nvars * tree_c1_nsplits * tree_c2_nvars * tree_c2_nsplits))

        # compute MH info data
        self.calc_change_swap_mh_data(transition_p=transition_p) 

        return new_tree

        
    def calc_grow_prune_mh_data(self, new_tree: Tree, node_to_split: Node, 
                          tot_parents_with_two_leaves: int, b: int):
        """
        Compute and store move-specific information needed for the grow/prune move acceptance ratio.

        Parameters
        ----------
        new_tree : Tree
            The proposed new tree.
        node_to_split : Node
            The node that was split (or pruned).
        tot_parents_with_two_leaves : int
            Total number of internal nodes with two leaves.
        b : int
            The number of leaves in the current tree.
        """
        raise AbstractMethodError()
    
    def calc_change_swap_mh_data(self, transition_p: float):
        """
        Compute and store move-specific information needed for the change/swap move acceptance ratio.

        Parameters
        ----------
        transition_p : float
            The transition probability ratio for the move.
        """
        raise AbstractMethodError()

    def update_tree(self, move: str):
        """
        Evolve the tree based on the specified move.

        Parameters
        ----------
        move : str
            The move type ('grow', 'prune', 'change', 'swap').

        Returns
        -------
        Tree
            The new tree after applying the move.
        """
        if move == 'grow':
            new_tree = self.grow_copy()
        elif move == 'prune':
            new_tree = self.prune_copy()
        elif move == 'change':
            new_tree = self.change_copy()
        elif move == 'swap':
            new_tree = self.swap_copy()

        if self.debug:
            if not new_tree.is_valid():
                raise ValueError('not-a-tree returned, BUG!!')
            
        # if the tree has been successfully updated, remove cached info
        new_tree.llik = None
        new_tree.log_tree_prior_prob = None
            
        return new_tree

        

    def sample_leaf_params(self) -> tuple[Any, Any]:
        """
        Sample new parameters for the leaves from the prior full conditionals.

        Returns
        -------
        tuple
            (left_leaf_params, right_leaf_params)
        """
        if self.is_classification:
            l_probs = self.sample_prior_p()
            r_probs = self.sample_prior_p()
            return (l_probs, r_probs)
        else:
            l_sigma = self.sample_prior_sigma()
            l_mu = self.sample_prior_mu(l_sigma)
            r_sigma = self.sample_prior_sigma()
            r_mu = self.sample_prior_mu(r_sigma)
            return (l_mu, l_sigma), (r_mu, r_sigma)
        
    def resample_leaf_params(self):
        """
        Resample the leaf parameters from their full conditionals.
        """
        if self.is_classification:
            self.resample_p()
        else:
            self.resample_mu_sigma()

    def resample_p(self):
        """
        Resample class probabilities for each leaf.
        """
        for leaf in self.tree.get_leaves():
            param = self.get_posterior_params_p(leaf)
            new_p = self.rng.dirichlet(param)
            leaf.update_node_params(new_p)

    def get_posterior_params_p(self, leaf: Node) -> NDArrayFloat:
        """
        Get the posterior Dirichlet parameters for a leaf (classification).

        Parameters
        ----------
        leaf : Node

        Returns
        -------
        NDArrayFloat
            The updated Dirichlet parameters.
        """
        n_k = leaf.count_values()
        return self.alpha_dirichlet + n_k

    def get_posterior_params_sigma(self, leaf: Node) -> tuple[float, float]:
        """
        Get the posterior parameters for sigma for a leaf (regression).

        Parameters
        ----------
        leaf : Node

        Returns
        -------
        tuple
            (a, scale) parameters for the inverse gamma full conditional.
        """
        idx, leaf_preds = leaf.get_preds()
        _idx, leaf_truth = leaf.get_true_preds()
        n_obs = leaf.get_nobs()

        if self.debug:
            assert np.all(idx == _idx)

        # un-normalized leaf MSE
        tot_se: float = np.sum((leaf_preds - leaf_truth)**2)

        a =  (n_obs+self.nu)/2
        scale = (tot_se + self.nu*self.lambd)/2
        return a, scale

    def get_posterior_params_mu(self, leaf: Node, new_sigma: float) -> tuple[float, float]:
        """
        Get the posterior parameters for mu for a leaf (regression).

        Parameters
        ----------
        leaf : Node
            The leaf node.
        new_sigma : float
            The resampled sigma value.

        Returns
        -------
        tuple
            (m, std) for the normal full conditional.
        """
        _idx, leaf_truth = leaf.get_true_preds()
        n_obs = leaf.get_nobs()

        # resample mu, compute parameters of full conditional
        m = (leaf_truth.sum() + self.a * self.mu_bar)/(n_obs + self.a)
        std = new_sigma/np.sqrt(n_obs + self.a)
        return m, std
    
    def resample_mu_sigma(self):
        """
        Resample mu and sigma for each leaf (regression).
        """
        for leaf in self.tree.get_leaves():
            # compute parameters of sigma's full conditional
            a, scale = self.get_posterior_params_sigma(leaf)
            # resample sigma
            new_sigma = np.sqrt(scipy.stats.invgamma.rvs(a=a, scale=scale, random_state=self.rng))

            # compute parameters of mu's full conditional
            m, std = self.get_posterior_params_mu(leaf, new_sigma)
            # resample mu
            new_mu = self.rng.normal(m, std, 1)[0]

            leaf.update_node_params((new_mu, new_sigma))
    
    def resample_mu_sigma_old(self):
        """
        (Deprecated) Resample mu and sigma using the old full conditional formulas.
        
        Raises
        ------
        Exception
            Always raises an exception indicating deprecation.
        """
        for leaf in self.tree.get_leaves():
            # current model prediction (posterior mean), to be resampled
            idx, leaf_preds = leaf.get_preds()
            _idx, leaf_truth = leaf.get_true_preds()
            n_obs = leaf.get_nobs()

            if self.debug:
                assert np.all(idx == _idx)

            # un-normalized leaf MSE
            tot_se = np.sum((leaf_preds - leaf_truth)**2)

            # resample sigma
            new_sigma = np.sqrt(scipy.stats.invgamma.rvs(a=(n_obs+self.nu)/2, scale=(tot_se + self.nu*self.lambd)/2, random_state=self.rng))

            # resample mu, compute parameters of full conditional
            m = (leaf_truth.sum() + self.a * self.mu_bar)/(n_obs + self.a)
            std = new_sigma/np.sqrt(n_obs + self.a)
            new_mu = self.rng.normal(m, std, 1)[0]

            leaf.update_node_params((new_mu, new_sigma))
        raise Exception('DEPRECATED')

    def get_log_posterior_prob(self, tree: Tree) -> float:
        """
        Compute the log posterior probability of a tree (up to a normalizing constant).

        This includes the full conditionals for leaf parameters, the tree prior, and the integrated likelihood.

        P(\theta, T | X, Y) \\prop P(\theta | T, X, Y) * P(T | X, Y)
        P(\theta | T, X, Y) = P(mu | T, X, Y, sigma) * P(sigma | T, X, Y)
        P(T | X, Y) \\prop P(Y | T, X) * P(T | X)

        Parameters
        ----------
        tree : Tree

        Returns
        -------
        float
            The log posterior probability.
        """
        log_lik_post_params = 0
        for leaf in tree.get_leaves():
            if self.is_classification:
                alpha = self.get_posterior_params_p(leaf)
                quantiles = leaf.get_params()
                log_lik_post_params += scipy.stats.dirichlet.logpdf(quantiles, alpha)
            else:
                mu, sigma = leaf.get_params()
                a, scale = self.get_posterior_params_sigma(leaf)
                log_lik_post_params += scipy.stats.invgamma.logpdf(sigma**2, a=a, scale=scale)
                m, std = self.get_posterior_params_mu(leaf, sigma)
                log_lik_post_params += scipy.stats.norm.logpdf(mu, m, std)

        # log(P(T | X))
        log_tree_prior_prob = self.calc_log_tree_prob(tree)

        # log(P(Y | T, X))
        log_lik_tree = self.calc_llik(tree)

        posterior = log_lik_post_params + log_tree_prior_prob + log_lik_tree
        return posterior
    
    def get_p_split(self, depth: int) -> float:
        """
        Compute the probability of splitting a node at a given depth.

        Parameters
        ----------
        depth : int
            The node depth.

        Returns
        -------
        float
            The splitting probability.
        """
        return self.alpha/(1+depth)**self.beta
     
    def calc_log_tree_prob(self, tree: Tree) -> float:
        """
        Compute the log prior probability of the tree structure P(T|X). 
        
        This is the prob of sampling a specific tree given a dataset x

        Parameters
        ----------
        tree : Tree

        Returns
        -------
        float
            The log prior probability.
        """

        def _calc_rec(node: Node, tot: float) -> float:
            assert not node.is_leaf()

            split_vars, split_vals, is_cat_split = node.get_split_info()
            avail_splits,_ = node.get_available_splits()
            n_vars = len(avail_splits)
            n_vals = len(avail_splits[split_vars])

            if is_cat_split:
                # compute all possible combinations
                if n_vals != 1:
                    n = n_vals - 1
                    n_vals = sum(list(map(lambda x: math.comb(n+1, x), range(1, n+1))))

            # current node is internal, update probability
            depth = node.depth
            tot += np.log(self.get_p_split(depth)) - (np.log(n_vars) + np.log(n_vals))

            # if child is terminal, update probability and return
            l_child, r_child = tree.get_children(node)
            if l_child.is_leaf():
                tot += np.log((1-self.get_p_split(depth+1)))
            else:
                tot = _calc_rec(l_child, tot)

            if r_child.is_leaf():
                tot += np.log((1-self.get_p_split(depth+1)))
            else:
                tot = _calc_rec(r_child, tot)

            return tot
        
        if tree.log_tree_prior_prob is not None:
            log_tree_prior_prob = tree.log_tree_prior_prob
            self.cache_counters['log_tree_prior_prob_cached'] += 1

            if self.debug:
                if tree.is_stump():
                    log_tree_prior_prob = np.log(1-self.get_p_split(0))
                else:
                    log_tree_prior_prob = _calc_rec(tree.get_root(), 0)
                assert log_tree_prior_prob == tree.log_tree_prior_prob
        else:
            if tree.is_stump():
                log_tree_prior_prob = np.log(1-self.get_p_split(0))
            else:
                log_tree_prior_prob = _calc_rec(tree.get_root(), 0)
        
        tree.log_tree_prior_prob = log_tree_prior_prob
        self.cache_counters['log_tree_prior_prob_tot'] += 1
        return log_tree_prior_prob
                  

    
class BCARTClassicSlow(BCART):
    """
    A slower implementation of BCART following the original CGM98 algorithm.

    This can be made more efficient by improving underlying operations in the BCART class.
    """
    def trans_prob_log(self, move: str, new_tree: Tree, old_tree: Tree) -> float:
        if move in ['change', 'swap']:
            return 0
        
        # Process Grow and Prune. 
        # Compute only the former as the latter is just the inverse.
        alpha, beta = self.alpha, self.beta
        b = self.mh_move_data['b']
        depth = self.mh_move_data['depth']
        n_int_with_2_child = self.mh_move_data['n_int_with_2_child']

        res = np.log(b) + np.log(alpha) + 2* np.log(1-alpha/(2+depth)**beta) - np.log((1+depth)**beta - alpha) - np.log(n_int_with_2_child)

        if move == 'grow':
            return np.log(self.move_prob[1]/self.move_prob[0])+res
        else:
            return np.log(self.move_prob[0]/self.move_prob[1])-res
        
    def calc_grow_prune_mh_data(self, new_tree: Tree, node_to_split: Node, 
                          tot_parents_with_two_leaves: int, b: int):
        node_to_split_depth = node_to_split.depth
        self.mh_move_data = {'depth': node_to_split_depth, 'n_int_with_2_child': tot_parents_with_two_leaves, 'b': b}
        
    
    def calc_change_swap_mh_data(self, transition_p: float):
        pass
    




class BCARTFast(BCART):
    """
    Optimized version of BCART that uses the "fast" classes. 
    
    Optimizations:
    - Improved memory usage on copy operations ("fast" classes)
    - Using custom sampling functions instead of scipy's for speed
    Optimized for memory.
    """
    @wraps(BCART.__init__) 
    def __init__(self, *args, **kwargs):
        self.node_data_reg = NodeDataRegressionFast
        self.node_data_class = NodeDataClassificationFast
        self.tree_class = TreeFast
        super().__init__(*args, **kwargs)
    __init__.__doc__ = BCART.__init__.__doc__
    # needed for Sphinx to generate the __init__ signature according to the parent class BCART, since we are roverloading the __init__
    # __init__.__signature__ = signature(BCART.__init__)
    
    def sample_prior_sigma(self) -> float:
        if self.debug:
            assert isinstance(self.nu, float) and isinstance(self.lambd, float)
        # a = shape
        return np.sqrt(invgamma_rvs(a=self.nu/2, scale=self.nu*self.lambd/2, rng=self.rng))
    
    def resample_mu_sigma(self):
        for leaf in self.tree.get_leaves():
            # compute parameters of sigma's full conditional
            a, scale = self.get_posterior_params_sigma(leaf)
            # resample sigma
            new_sigma = np.sqrt(invgamma_rvs(a=a, scale=scale, rng=self.rng))

            # compute parameters of mu's full conditional
            m, std = self.get_posterior_params_mu(leaf, new_sigma)
            # resample mu
            new_mu = self.rng.normal(m, std, 1)[0]

            leaf.update_node_params((new_mu, new_sigma))
    
    def get_log_posterior_prob(self, tree: Tree) -> float:
        '''
        Compute posterior probability for a tree (up to norming constant)
        P(\theta, T | X, Y) \\prop P(\theta | T, X, Y) * P(T | X, Y)
        P(\theta | T, X, Y) = P(mu | T, X, Y, sigma) * P(sigma | T, X, Y)
        P(T | X, Y) \\prop P(Y | T, X) * P(T | X)
        '''
        log_lik_post_params = 0
        for leaf in tree.get_leaves():
            if self.is_classification:
                alpha = self.get_posterior_params_p(leaf)
                quantiles = leaf.get_params()
                log_lik_post_params += dirichlet_logpdf(quantiles, alpha)
            else:
                mu, sigma = leaf.get_params()
                a, scale = self.get_posterior_params_sigma(leaf)
                log_lik_post_params += invgamma_logpdf(sigma**2, a=a, scale=scale)
                m, std = self.get_posterior_params_mu(leaf, sigma)
                log_lik_post_params += norm_logpdf(mu, m, std)

        # log(P(T | X))
        log_tree_prior_prob = self.calc_log_tree_prob(tree)

        # log(P(Y | T, X))
        log_lik_tree = self.calc_llik(tree)

        posterior = log_lik_post_params + log_tree_prior_prob + log_lik_tree
        return posterior
    
    def update_tree(self, move: str):
        if move == 'grow':
            new_tree = self.grow_copy(light=True)
        elif move == 'prune':
            new_tree = self.prune_copy(light=True)
        elif move == 'change':
            new_tree = self.change_copy(light=True)
        elif move == 'swap':
            new_tree = self.swap_copy(light=True)

        if self.debug:
            if not new_tree.is_valid():
                raise ValueError('not-a-tree returned, BUG!!')
            
        # if the tree has been successfully updated, remove cached info
        new_tree.llik = None
        new_tree.log_tree_prior_prob = None
            
        return new_tree




class BCARTClassic(BCARTFast, BCARTClassicSlow):
    """
    BCART implementation using the classic CGM98 algorithm.
    """
    pass
