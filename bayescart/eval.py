"""
Utility functions for BCART experiments evaluation and analysis.

This module contains helper functions for computing tree probabilities,
comparing trees from BCART runs and provide summary tables of estimated 
posterior probabilities for the most commonly visited trees. This can be
used to gauge convergence. 

"""

from .tree import Tree
from .bcart import BCART
import numpy as np
import pandas as pd

def _gen_bcart_obj(tree: Tree, run):
    """
    Generate a BCARTClassic object from a given tree and run result. Can be used to compute probabilities.

    This function makes a copy of the input tree, assigns the data from the run
    (X and y) to the root node if not already present, updates the subtree data,
    and returns a BCARTClassic object configured with the run setup.

    Parameters
    ----------
    tree : Tree
        The tree for which to generate a BCART object.
    run : dict
        A run result dictionary containing keys 'data' and 'setup'. The 'data'
        key must include X and y; 'setup' includes hyperparameters such as alpha,
        beta, a, mu_bar, nu, lambd, iters, burnin, thinning, and move_prob.

    Returns
    -------
    BCARTClassic
        A BCARTClassic object initialized with the given tree and run parameters.
    """
    tree = tree.copy()
    X, y = run['data']['X'], run['data']['y']
    stp = run['setup']
    if not tree.nodes[0].has_data():
        tree.get_root()._data.X = run['data']['X']
        tree.get_root()._data.y = run['data']['y']
        tree.update_subtree_data(tree.get_root())

    bcart = BCART(X=X, y=y, alpha=stp['alpha'], beta=stp['beta'], a=stp['a'], mu_bar=stp['mu_bar'], nu=stp['nu'], lambd=stp['lambd'], iters=stp['iters'], burnin=stp['burnin'], thinning=stp['thinning'], move_prob=stp['move_prob'], light=stp['light'], seed=stp['seed'])
    bcart.tree = tree
    return bcart


def calc_tree_post_prob(tree: Tree, run):
    """
    Calculate the (log) posterior probability of a tree given a run result.

    Parameters
    ----------
    tree : Tree
        The tree for which to compute the posterior probability.
    run : dict
        A run result dictionary containing configuration and data information.

    Returns
    -------
    float
        The log posterior probability of the given tree.
    """
    bcart = _gen_bcart_obj(tree, run)
    return bcart.get_log_posterior_prob(tree)

def calc_tree_llik(tree: Tree, run):
    """
    Calculate the integrated log-likelihood of a tree given a run result.

    Parameters
    ----------
    tree : Tree
        The tree for which to compute the likelihood.
    run : dict
        A run result dictionary containing configuration and data information.

    Returns
    -------
    float
        The integrated log-likelihood P(Y|X,T) of the tree.
    """
    if tree.llik is not None:
        return tree.llik
    bcart = _gen_bcart_obj(tree, run)
    return bcart.calc_llik(bcart.tree)

def calc_log_tree_prob(tree: Tree, run):
    """
    Calculate the log prior probability of a tree given a run result.

    Parameters
    ----------
    tree : Tree
        The tree for which to compute the prior probability.
    run : dict
        A run result dictionary containing configuration and data information.

    Returns
    -------
    float
        The log prior probability P(T|X) of the tree.
    """
    if tree.log_tree_prior_prob is not None:
        return tree.log_tree_prior_prob
    bcart = _gen_bcart_obj(tree, run)
    return bcart.calc_log_tree_prob(bcart.tree)

def calc_log_tree_prob_and_llik(tree: Tree, run):
    """
    Calculate both the log prior probability and the integrated log-likelihood of a tree.

    Parameters
    ----------
    tree : Tree
        The tree for which to compute the probabilities.
    run : dict
        A run result dictionary containing configuration and data information.

    Returns
    -------
    tuple of float
        A tuple (prior, llik) where 'prior' is the log prior probability and 'llik'
        is the integrated log-likelihood of the tree.
    """
    if tree.llik is not None:
        llik =  tree.llik
    else:
        llik = None
    if tree.log_tree_prior_prob is not None:
        prior = tree.log_tree_prior_prob
    else:
        prior = None
    if llik is None or prior is None:
        bcart = _gen_bcart_obj(tree, run)
        if llik is None:
            llik = bcart.calc_llik(bcart.tree)
            tree.llik = llik
        if prior is None:
            prior = bcart.calc_log_tree_prob(bcart.tree)
            tree.log_tree_prior_prob = prior
    return prior, llik



#%% Compare trees rigorously

def _get_run_from_res(res_or_run):
    if isinstance(res_or_run, list):
        return res_or_run[0]
    elif isinstance(res_or_run, dict):
        return res_or_run
    
def _calc_cats_if_data_missing(res_or_run):
    run = _get_run_from_res(res_or_run)
    if not run['tree_store'][0].all_nodes()[0].has_data(): # type: ignore
        categories = {}
        X, y = run['data']['X'], run['data']['y'] # type: ignore
        for col in X.columns:
            if hasattr(X[col], 'cat'):
                categories[col] = set(X[col].cat.categories)
        return categories
    else:
        return None

def compare_trees(tree1: Tree, tree2: Tree, res_or_run, type='prob'):
    """
    Compare two trees to check if they are equivalent under a given mode.

    There are different levels of comparison:
    - Basic, using buil-in tree comaprison (hard=0)
    - Probability, using data likelihood and tree prior (hard=1). 
    Note that different likelihoods imply different partitions, 
    with high probability. Instead, the prior should tell apart trees 
    that have same partition but different structure so that the two 
    trees are not probabilistically equivalent.

    Parameters
    ----------
    tree1 : Tree
        The first tree to compare.
    tree2 : Tree
        The second tree to compare.
    res_or_run : dict or list
        Either a single run result dictionary or a list of run results,
        used to extract necessary configuration and data.
    type : str, optional
        Comparison type ('basic' or 'prob'). Default is 'prob'.

    Returns
    -------
    bool
        True if the trees are considered equal under the chosen criteria, False otherwise.
    """

    categories = _calc_cats_if_data_missing(res_or_run)
    basic = tree1.is_equal(tree2, hard=1, categories=categories)
    if basic:
        # if this says they are the same, then they are. If No, it might miss some.
        return True
    if type == 'basic':
        return basic
    if type == 'prob':
        prior1, llik1 = calc_log_tree_prob_and_llik(tree1, _get_run_from_res(res_or_run))
        prior2, llik2 = calc_log_tree_prob_and_llik(tree2, _get_run_from_res(res_or_run))

        # bcart1 = _gen_bcart_obj(tree1, _get_run_from_res(res_or_run))
        # bcart2 = _gen_bcart_obj(tree2, _get_run_from_res(res_or_run))
        # llik1 = bcart1.calc_llik(bcart1.tree)
        # llik2 = bcart2.calc_llik(bcart2.tree)
        # prior1 = bcart1.calc_log_tree_prob(bcart1.tree)
        # prior2 = bcart2.calc_log_tree_prob(bcart2.tree)
        cond1 = np.isclose(llik1, llik2)
        cond2 = np.isclose(prior1, prior2)
        prob = cond1 and cond2
        return prob
    
    
def _find_tree_idx(tree, trees, res_or_run, type):
    """
    Find the index of a tree within a list of unique trees.

    Parameters
    ----------
    tree : Tree
        The tree to search for.
    trees : list
        A list of unique Tree objects.
    res_or_run : dict or list
        A run result dictionary or list of run results.
    type : str
        Comparison type for matching trees (e.g. 'prob' or 'basic').

    Returns
    -------
    int or None
        The index of the matching tree if found; otherwise, None.
    """
    for idx, unq_tree in enumerate(trees):
        if compare_trees(tree, unq_tree, res_or_run, type=type):
            return idx
    return None


def _summarize_trees(trees, res_or_run, top_n: int|None = None, type='prob'):
    tot_trees = len(trees)
    unique_trees: list[Tree] = [trees[0]]
    counts = {0: 1}
    for tree in trees[1:]:
        tree_idx = _find_tree_idx(tree, unique_trees, res_or_run, type)
        if tree_idx is not None:
            counts[tree_idx] += 1
        else:
            unique_trees.append(tree)
            counts[len(unique_trees)-1] = 1
    most_freq_trees_idx = sorted([(k,v/tot_trees) for k, v in counts.items()], key=lambda x: x[1], reverse=True)
    if top_n is not None:
        most_freq_trees_idx = most_freq_trees_idx[:top_n]
    else:
        top_n = len(unique_trees)
    return [most_freq_trees_idx[i][1] for i in range(top_n)], [unique_trees[idx[0]] for idx in most_freq_trees_idx]


def summarize_trees(run, top_n=None, type='prob', plot=False):
    """
    Summarize the most common trees from a single run.

    The function extracts the stored 'tree_store' from the run result, summarizes the top
    trees using frequency of occurrence (by the chosen comparison type), and optionally plots
    each unique tree.

    Parameters
    ----------
    run : dict
        A run result dictionary containing a 'tree_store' key.
    top_n : int or None, optional
        The number of top trees to extract. If None, all unique trees are returned.
    type : str, optional
        The type of comparison to use ('prob' or 'basic'). Default is 'prob'.
    plot : bool, optional
        If True, each unique tree is displayed using its show() method. Default is False.

    Returns
    -------
    tuple
        A tuple (freq_list, unique_trees) representing the frequencies and corresponding unique trees.
    """
    trees: list[Tree] = run['tree_store']
    res = _summarize_trees(trees, run, top_n, type)
    if plot:
        for tree in res[1]:
            tree.show()
    return res


def produce_tree_table(res):
    """
    Produce a summary table of tree posterior probability estimates across runs.

    This function:
      1. Extracts the top 5 most frequent trees from each run.
      2. Pools all top trees and computes a unique set.
      3. Sorts the unique trees by their average empirical frequency (posterior probability)
         across runs.
      4. Constructs a table with each unique tree's estimated frequency per run, mean, standard
         deviation, and number of terminal nodes.

    Parameters
    ----------
    res : list
        A list of run result dictionaries.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame summarizing the frequency estimates for the most frequent trees.
    """

    # 1. extract the top 5 most likely from each run
    freq_runs = []
    unique_trees_runs = []
    all_trees = []
    for run in res:
        freq, unique_trees = summarize_trees(run)
        freq_runs.append(freq)
        unique_trees_runs.append(unique_trees)

        # now only keep the top 5 trees
        all_trees.extend(unique_trees[:5])
    
    # 2. we pool all top 5's and take the unique set; this number gives us the total rows
    _, unique_trees = _summarize_trees(all_trees, res)
    nrows = len(unique_trees)
    ncols = len(res)
    tbl = np.zeros((nrows, ncols))
    # for each row, we want to find the corresponding chain probability (col), hence compare prob across all visited trees
    for col in range(ncols):
        freq, trees = freq_runs[col], unique_trees_runs[col]
        for row in range(nrows):
            # find the frequency in the relevant chain fo the current row tree
            tree_idx = _find_tree_idx(unique_trees[row], trees, res, type='prob')
            tbl[row, col] = float(f'{freq[tree_idx]:0.2f}') if tree_idx is not None else 0  
    
    sorted_idx = np.argsort(tbl.mean(axis=1))[::-1]
    tbl = tbl[sorted_idx,:]
    n_leaves = np.array([unique_trees[idx].get_n_leaves() for idx in sorted_idx])
    mn = np.round(tbl.mean(axis=1)[:,np.newaxis], decimals=2)
    std = np.round(tbl.std(axis=1)[:,np.newaxis], decimals=2)
    tbl = np.hstack([tbl, mn, std, n_leaves[:,np.newaxis]])
    tbl = pd.DataFrame(tbl, columns=[f'C{i}' for i in range(ncols)] + ['Mn', 'Std', 'b'])
    return tbl