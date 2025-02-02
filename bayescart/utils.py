"""Utility functions for bayescart.

This module provides helper functions for sampling from distributions,
computing log-probability densities, and a generic choice function.
"""

from typing import Sequence
import numpy as np
from scipy.special import gammaln, gammainccinv
import matplotlib.pyplot as plt
import pandas as pd


def my_choice[elem](rng: np.random.Generator, a: Sequence[elem], replace: bool = False, p: Sequence[float]|None = None) -> elem:
    """
    Sample a random element from a generic sequence using the provided random generator.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator.
    a : Sequence[elem]
        The sequence to sample from.
    replace : bool, optional
        Whether the sampling is done with replacement (default is False).
    p : Sequence[float] or None, optional
        The probability weights associated with each element (default is None).

    Returns
    -------
    elem
        A randomly selected element from the sequence.
    """
    sampled_idx = rng.choice(len(a), replace=replace, p=p)
    return a[sampled_idx]

def invgamma_rvs(a, scale, rng):
    """
    Sample a random variate from an inverse gamma distribution.

    Parameters
    ----------
    a : float
        The shape parameter (adjusted by 1/2 in this context).
    scale : float
        The scale parameter.
    rng : np.random.Generator
        The random number generator.

    Returns
    -------
    float
        A random variate from the inverse gamma distribution, scaled accordingly.
    """
    U = rng.uniform()
    Y = 1.0 / gammainccinv(a, U)
    return Y * scale

def invgamma_logpdf(x, a, scale):
    """
    Compute the log probability density function of an inverse gamma distribution.

    Parameters
    ----------
    x : float
        The point at which to evaluate the log-pdf.
    a : float
        The shape parameter.
    scale : float
        The scale parameter.

    Returns
    -------
    float
        The log probability density.
    """
    return a*np.log(scale) - gammaln(a) - (a+1)*np.log(x) - scale/x

def norm_logpdf(x, a, scale):
    """
    Compute the log probability density function of a normal distribution.

    Parameters
    ----------
    x : float
        The point at which to evaluate the log-pdf.
    a : float
        The mean of the distribution.
    scale : float
        The standard deviation of the distribution.

    Returns
    -------
    float
        The log probability density.
    """
    return -0.5*np.log(2*np.pi) - np.log(scale) - 0.5*((x-a)/scale)**2

def dirichlet_logpdf(x, alpha):
    """
    Compute the log probability density function of a Dirichlet distribution.

    Parameters
    ----------
    x : array-like
        The point (vector) at which to evaluate the log-pdf.
    alpha : array-like
        The concentration parameters of the Dirichlet distribution.

    Returns
    -------
    float
        The log probability density.
    """
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((alpha-1)*np.log(x))


##### Plotting utils #####


def _plot_hist(data, title='', ylbl='', xlbl=''):
    '''Data is a list of comparable distributions. For each, we calculate the histogram and place them side by side, with common x-axis.'''
    nchains = data.shape[0]
    fig, ax = plt.subplots(1, nchains)
    for i in range(nchains):
        ax[i].hist(data[i], bins=np.arange(data.min()-0.5, data.max()+0.5, 1), orientation='horizontal', density=True) # type: ignore
        if i != 0:
            ax[i].get_yaxis().set_visible(False)  # type: ignore
    fig.suptitle(title)
    ax[0].set_ylabel(ylbl) # type: ignore
    fig.supxlabel(xlbl) # type: ignore
    fig.tight_layout()

def plot_hists(res, idx_range=(0,1), title=''):
    """
    Plot the distribution of terminal nodes for the cold chain across runs.

    The function extracts the stored terminal node counts from each run in 'res'
    and plots histograms for a selected fraction of the post-burnin iterations,
    where idx_range is given as (min_percentage, max_percentage).

    Parameters
    ----------
    res : list or sequence of dict
        A list of run result dictionaries, each containing a 'setup' key and a
        'tree_term_reg' key.
    idx_range : tuple, optional
        A tuple (min_percentage, max_percentage) specifying the range (as fractions)
        of the available post-burnin observations to use for plotting. Default is (0,1).
    title : str, optional
        The title of the plot (default is '').

    Returns
    -------
    None
    """
    # idx_range is expected to be a percentage, capturing time after burnin.
    # This makes it independent of the length, and comparable across algos
    min_idx_perc, max_idx_perc = idx_range
    setup = res[0]['setup']
    iters, burnin, thinning = setup['iters'], setup['burnin'], setup['thinning']

    available_obs = (iters - burnin) / thinning
    min_idx = int(min_idx_perc * available_obs)
    max_idx = int(max_idx_perc * available_obs)

    term_reg = np.array([res[i]['tree_term_reg'] for i in range(len(res))])
    # print(f'total obs: {term_reg.shape[1]}, predicted obs: {available_obs}, using from step {min_idx} to {max_idx} for plotting')
    print(f'Using from step {min_idx} to {max_idx} for plotting')
    _plot_hist(term_reg[:, min_idx:max_idx], title=title, ylbl='Number of terminal Regions', xlbl='Target (cold) chain ---> flatteer chains')


def plot_chain_comm(res, title=''):
    """
    Plot the distribution of terminal regions across the chains from a PT run.

    This function uses the stored parallel tempering (PT) statistics (the number
    of terminal nodes per chain) from the run results and plots them as histograms.
    It also prints the final swap acceptance probabilities.

    Parameters
    ----------
    res : dict
        A run result dictionary containing 'PT_swap_stats' with key 'PT_term_reg'
        and 'PT_swap_final_prob'.
    title : str, optional
        The title of the plot (default is '').

    Returns
    -------
    None
    """
    print(res['PT_swap_stats']['PT_swap_final_prob'])
    _plot_hist(res['PT_swap_stats']['PT_term_reg'], title=title, ylbl='Number of terminal Regions', xlbl='Target (cold) chain ---> flatteer chains')
    

def plot_swap_prob(res, title=''):
    """
    Plot the evolution of swap acceptance probabilities over time for PT chains.

    Parameters
    ----------
    res : dict
        A run result dictionary containing 'PT_swap_stats' with key
        'PT_swap_prob_over_time', which is a mapping from iteration numbers to swap
        acceptance probabilities.
    title : str, optional
        The title of the plot (default is '').

    Returns
    -------
    None
    """
    data = res['PT_swap_stats']['PT_swap_prob_over_time']
    xs = []
    ys = []
    labels = [f'{i[0]} (t= {i[1]})' for i in zip(range(len(data)), res['setup']['temps'])]
    for k,v in data.items():
        xs.append(k)
        ys.append(v)
    xs = np.array(xs)
    ys = np.array(ys)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, label=labels)
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Swap Acceptance Probability')
    ax.set_title(title)


def sim_cgm98(n, rng):
    """
    Simulate data following the CGM98 example.

    Parameters
    ----------
    n : int
        Number of observations.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame with features 'v1' and 'v2', and y is a Series.
    """
    v1 = rng.choice(np.arange(1, 11) - 0.5, size=n, replace=True)
    v2 = pd.Series(rng.choice(['a', 'b', 'c', 'd'], size=n, replace=True), dtype="category")
    e = 2 * rng.standard_normal(n)
    
    r1 = (v1 <= 5) & (v2.isin(['a', 'b']))
    r2 = (v1 > 5) & (v2.isin(['a', 'b']))
    r3 = (v1 <= 3) & (v2.isin(['c', 'd']))
    r4 = (v1 > 3) & (v1 <= 7) & (v2.isin(['c', 'd']))
    r5 = (v1 > 7) & (v2.isin(['c', 'd']))
    
    y = np.empty(n)
    y[:] = np.nan
    y[r1] = 8 + e[r1]
    y[r2] = 2 + e[r2]
    y[r3] = 1 + e[r3]
    y[r4] = 5 + e[r4]
    y[r5] = 8 + e[r5]
    
    X = pd.DataFrame({'v1': v1, 'v2': v2})
    y = pd.Series(y)
    return X, y
