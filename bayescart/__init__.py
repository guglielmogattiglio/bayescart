"""bayescart: A Python package for Bayesian CART with tempering.

This package provides classes and functions to build, sample, and evaluate
Bayesian CART models with advanced tempering strategies. The package is composed
of modules for node data handling, tree structures, Bayesian CART sampling, and
utility functions.

Available objects:
  - BCARTClassic, BCARTPT, BCARTGeom, BCARTGeomLik, BCARTPseudoPrior 
  - Tree 
  - Node 
  - NodeData, NodeDataRegression, NodeDataClassification 
  - Utility functions and plotting functions
  - Evaluation and functions for analysis
  - Exceptions: InvalidTreeError, AbstractMethodError 
"""

__version__ = "0.1.0"

from .bcart import BCARTClassic
from .tempering_geometric import BCARTGeom
from .tempering_geom_lik import BCARTGeomLik
from .tempering_pseudo_prior import BCARTPseudoPrior
from .tree import TreeFast as Tree
from .node import NodeFast as Node
from .node_data import NodeDataFast as NodeData
from .node_data import NodeDataRegressionFast as NodeDataRegression
from .node_data import NodeDataClassificationFast as NodeDataClassification
from .utils import (
    plot_hists,
    plot_chain_comm,
    plot_swap_prob,
    sim_cgm98,
)
from .eval import (
    calc_tree_post_prob,
    calc_tree_llik,
    calc_log_tree_prob,
    calc_log_tree_prob_and_llik,
    compare_trees,
    summarize_trees,
    produce_tree_table
)
