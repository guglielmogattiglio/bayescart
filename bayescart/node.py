"""Node class for bayescart.

This module defines the Node class used to represent nodes in a tree.
It extends the Node implementation from treelib.
"""

import numpy as np
from treelib import Node as TreelibNode
import pandas as pd
from typing import Sequence, Any
from copy import deepcopy
from .mytyping import NDArrayInt, NDArrayFloat, T
from .exceptions import InvalidTreeError
from .node_data import NodeData
from collections import defaultdict


class Node(TreelibNode):
    """
    Extended node class for Bayesian CART. Provides functionalities for adding data, splitting, and updating parameters.

    This class is mostly a wrapper around the NodeData object, which handles the data and parameters associated with the node.

    Attributes
    ----------
    is_l : bool
        Flag indicating if this node is a left child.
    _data : NodeData
        The node data (parameters and associated data).
    _rng : np.random.Generator
        The random number generator.
    debug : bool
        If True, enable debug checks.
    _depth : int
        The depth of the node.
    """
    def __init__(self, id: int, is_l: bool, data: NodeData, rng: np.random.Generator, debug: bool):
        super().__init__(identifier=id)
        self.is_l: bool = is_l
        self._data: NodeData = data
        self._rng = rng
        self.debug = debug
        self._depth = -1

    @property
    def id(self):
        return self.identifier

    @property
    def depth(self):
        return self._depth
    
    @depth.setter
    def depth(self, val: int):
        if val < 0:
            raise ValueError('Node depth must be non-negative')
        self._depth = val

    def __deepcopy__(self, memo):
        return self.copy(light=False, memo=memo)

    def copy(self, light: bool = False, no_data: bool = False, memo: dict|None = None) -> 'Node':
        if memo is None:
            memo = {}
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k == '_data':
                setattr(result, k, v.copy(light=light, no_data=no_data, memo=memo))
            elif k == '_rng':
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result
    
    def has_data(self) -> bool:
        return self._data.has_data()

    def _gen_tags(self):
        """
        Generate a string tag for the node for plotting purposes.
        """
        left_or_right = 'L' if self.is_l else 'R'
        if self.is_leaf():
            if self.has_data():
                self.tag = f"{left_or_right}_{self.identifier}_{self._data.get_nobs()}_{self._data.get_params(print=True)}"
            else:
                self.tag = f"{left_or_right}_{self.identifier}_{self._data.nobs}_{self._data.get_params(print=True)}" # type: ignore
        else:
            self.tag = f'{left_or_right}_{self.identifier}_{self._data.get_split_var(print=True)} {self._data.get_split_set(print=True)}'

    def get_nobs(self) -> int:
        return self._data.get_nobs()

    def get_available_splits(self, *args, **kw) -> tuple[dict[str, Sequence[T]], dict[str, bool]]:
        """
        Get available splits from the underlying NodeData.

        Returns
        -------
        tuple
            (avail_vars, is_cat)
        """
        return self._data.get_available_splits(*args, **kw)
    
    def get_new_split(self) -> tuple[str, Sequence[T] | T]:
        """
        Sample a new split for the node.

        Returns
        -------
        tuple
            (split_var, split_val)
        """
        # Note that the splitting procedure depends on whether the 
        # # variable is categorical or not. This is checked by the 
        # # NodeData object (which is the only object having access to the data).

        # Get all the possible splits
        avail_vars,_ = self._data.get_available_splits()

        # If no split available, return None
        if len(avail_vars) == 0:
            raise InvalidTreeError('No available variable to split')
        
        # sample a split variable uniformly
        split_var = self._rng.choice(list(avail_vars.keys()))
        avail_vals = avail_vars[split_var]

        # sample a split value
        is_cat, split_val = self._data.sample_split(split_var, avail_vals)

        return split_var, split_val
    
    def get_split_info(self) -> tuple[str, Sequence[T] | T, bool]:
        """
        Retrieve the current split rule: split variable (str), split value (array if cat, float else), whether the split is categorical.

        Returns
        -------
        tuple
            (split_var, split_val, is_cat)
        """
        split_var, split_val = self._data.get_split_var(), self._data.get_split_set()
        is_cat = self._data.is_cat_split
        return split_var, split_val, is_cat

    def update_split_info(self, split_var: str, split_val: Sequence[T] | T):
        self._data.update_split_info(split_var, split_val)
        # self._gen_tags()

    def update_split_data(self, X: pd.DataFrame, y: pd.Series):
        self._data.update_split_data(X, y)
    

    def get_split_data(self, split_var: str, split_val: Sequence[T] | T, left_params: Any, right_params: Any) -> tuple[NodeData, NodeData]:
        """
        Split the data at this node into two parts. Returns the children.

        Parameters
        ----------
        split_var : str
            The feature to split on.
        split_val : Sequence[T] or T
            The split rule.
        left_params : Any
            Parameters for the left child.
        right_params : Any
            Parameters for the right child.

        Returns
        -------
        tuple
            (left_node_data, right_node_data)
        """
        return self._data.get_split_data(split_var, split_val, left_params, right_params)
    
    def get_data_split(self, split_var: str|None = None, split_val: Sequence[T] | T|None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the node's data using the current (or provided) split rule.  Returns the left and right data subsets.

        Parameters
        ----------
        split_var : str or None, optional
            The splitting feature. If None, uses the current split_var.
        split_val : Sequence[T] or T or None, optional
            The splitting value(s). If None, uses the current split_set.

        Returns
        -------
        tuple
            (left_X, right_X, left_y, right_y)
        """
        if split_var is None:
            split_var = self._data.get_split_var()
        if split_val is None:
            split_val = self._data.get_split_set()
        return self._data.get_data_split(split_var, split_val)
    
    def is_split_rule_empty(self) -> bool:
        return self._data.is_split_rule_emtpy()
    
    def count_values(self) -> NDArrayInt:
        """
        Count the of observations per class.

        Returns
        -------
        NDArrayInt
            Array of counts per class.
        """
        return self._data.count_values()
    
    def get_data_averages(self) -> tuple[float, float, float]:
        """
        Compute data averages (number of observations, mean, un-normalized variance) for regression.

        Returns
        -------
        tuple
            (n, mean, un-normalized variance)
        """
        return self._data.get_data_averages()
        
    def update_node_params(self, params: tuple[float, float] | NDArrayFloat):
        self._data.update_node_params(params)
        # self._gen_tags()

    def get_preds(self) -> tuple[NDArrayFloat, float]:
        """
        Get the predictions from the node's model.

        Returns
        -------
        tuple
            (indices, predictions)
        """
        return self._data.get_preds()
    
    def get_true_preds(self) -> tuple[NDArrayFloat, NDArrayFloat]:
        """
        Return the true predictions (i.e. the actual response values).

        Returns
        -------
        tuple
            (indices, true response values)
        """
        y = self._data.y
        return y.index.to_numpy(), y.to_numpy()

    def get_params(self, print: bool = False) -> Any:
        return self._data.get_params(print)

    def calc_avail_split_and_vars(self) -> tuple[int, int]:
        """
        Compute the number of available variables for splitting, and the number of available splits for the current split variable.

        Returns
        -------
        tuple
            (number of available variables, number of available splits)
        """
        return self._data.calc_avail_split_and_vars()

    def reset_split_info(self):
        self._data.reset_split_info()



class NodeFast(Node):
    """
    Fast implementation of Node with optimized copy operations.
    """

    def copy(self, light: bool = False, no_data: bool = False, memo: dict|None = None) -> 'Node':
        """
        Copy the node, with an option for a light (optimized) copy.

        Change: Instead of thoroughly copying the object, just copy what we know we need.
        """ 
        
        # _initial_tree_id contains the hash (str) of the first tree the node was attached to. In our case, this is unique. 
        # _predecessor is a dict mapping tree_ids to node_ids, with the idea of sharing nodes across trees, potentially. 
        # I am still not sure about this implementation, but it doesn't harm me.
        # Similarly, _successors maps onto the list of children, which in my case is up to two. 
            
        if not light:
            return super().copy(light=light, no_data=no_data, memo=memo)
        else:
            cls = self.__class__
            result = cls.__new__(cls)
            result.is_l = self.is_l
            result._data = self._data.copy(light=light, no_data=no_data, memo=memo)
            result.debug = self.debug
            result.identifier = self.identifier
            result._depth = self._depth
            result._tag = self._tag
            result.data = self.data
            result.expanded = self.expanded
            result._rng = self._rng 
            result.ADD = self.ADD
            result.DELETE = self.DELETE
            result.INSERT = self.INSERT
            result.REPLACE = self.REPLACE

            k = self._initial_tree_id
            if self.debug:
                assert len(self._predecessor) == 1
                assert len(self._successors[k]) <= 2
            if k is not None:
                result._predecessor = {k: self._predecessor[k]}
                result._successors = self._successors
                result._successors = defaultdict(list)
                result._successors[k] = [x for x in self._successors[k]]
                result._initial_tree_id = k
            else:
                result._predecessor = {}
                result._successors = defaultdict(list)
                result._initial_tree_id = None

            return result

