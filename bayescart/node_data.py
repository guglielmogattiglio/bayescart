"""Node data classes for bayescart.

This module defines classes that encapsulate the data associated with
a tree node. It provides a base class NodeData along with subclasses for
regression (NodeDataRegression) and classification (NodeDataClassification).
"""

import numpy as np
import pandas as pd
from typing import Sequence, Self, Any
import math
from copy import deepcopy
from .mytyping import T, NDArrayInt, NDArrayFloat
from .exceptions import AbstractMethodError, InvalidTreeError
from functools import wraps

class NodeData():
    """
    Abstract base class for data associated with a node in the CART tree.
    
    Attributes
    ----------
    X : pd.DataFrame
        The data features at this node.
    y : pd.Series
        The response variable at this node.
    rng : np.random.Generator
        Random number generator for sampling.
    debug : bool
        If True, enables additional debugging checks.
    node_min_size : int
        Minimum number of observations required in the node.
    split_var : str
        The variable used for splitting (if any).
    split_set : Sequence[T] or T
        The value(s) used in the splitting rule.
    is_cat_split : bool
        Indicator whether the split is categorical.
    avail_splits : Any
        Cached available splits (initially None).
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, 
                 rng: np.random.Generator, 
                 debug: bool,
                 node_min_size: int,
                 split_var: str | None = None, 
                 split_set: Sequence[T] | T |None = None,
                 is_cat_split: bool | None = None):
        self.node_min_size = node_min_size
        self.X = X
        self.y = y
        self.rng = rng
        if (_tot := (is_cat_split is None) + (split_var is None) + (split_set is None)) != 0 and _tot != 3:
            raise ValueError('Either all or none of the split parameters must be None')
        self.split_var = split_var if split_var is not None else ''
        self.is_cat_split = is_cat_split if is_cat_split is not None else False
        self.split_set = split_set if split_set is not None else ""
        self.debug = debug
        self.avail_splits = None

    @property
    def X(self) -> pd.DataFrame:
        return self._X
    
    @X.setter
    def X(self, val: pd.DataFrame):
        if not isinstance(val, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')
        if self.get_nobs_from_arg(val) < self.node_min_size:
            raise InvalidTreeError('Node has less than min node size observations')
        self._X = val

    @property
    def y(self) -> pd.Series:
        return self._y
    
    @y.setter
    def y(self, val: pd.Series):
        if not isinstance(val, pd.Series):
            raise TypeError('y must be a pandas Series')
        if self.get_nobs_from_arg(val) < self.node_min_size:
            raise InvalidTreeError('Node has less than min node size observations')
        self._y = val

    def has_data(self) -> bool:
        return (hasattr(self, '_X')) and (hasattr(self, '_y'))
        
    def __deepcopy__(self, memo):
        return self.copy(light=False, memo=memo)

    def copy(self, light: bool = False, no_data: bool = False, memo: dict|None = None) -> 'NodeData':
        if memo is None:
            memo = {}
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if (k == '_X' or k == '_y'):
                if no_data:
                    result.nobs = self.get_nobs() # type: ignore
                    continue
                elif light:
                    setattr(result, k, v)
                else:
                    setattr(result, k, deepcopy(v, memo))
            elif k == 'rng':
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result
        
    @staticmethod
    def _is_categorical(x) -> bool:
        return hasattr(x, 'cat')
        

    def get_nobs(self) -> int:
        return self.get_nobs_from_arg(self.X)
    

    @staticmethod
    def get_nobs_from_arg(x: pd.DataFrame | pd.Series) -> int:
        """
        Get the number of observations from a DataFrame or Series.

        Parameters
        ----------
        x : pd.DataFrame or pd.Series

        Returns
        -------
        int
            The number of rows (observations).
        """
        return x.shape[0]

    @staticmethod
    def _print(x) -> float:
        return np.round(x, 2)  
    
    
    def get_split_var(self, print: bool = False) -> str:
        return self.split_var
    
    
    def get_split_set[T](self, print: bool = False) -> Sequence[T] | T:
        """
        Get the splitting value(s) for this node.

        Parameters
        ----------
        print : bool, optional
            If True, return a formatted string description (default is False).

        Returns
        -------
        Sequence[T] or T
            The splitting value(s) or a description thereof.
        """
        if not print:
            return self.split_set # type: ignore
        else:
            if self.is_cat_split:
                return f'in {self.split_set}' # type: ignore
            else:
                return f'< {self.split_set}' # type: ignore
            
            
    def get_available_splits(self, force_eval=False, assert_cached=False) -> tuple[dict[str, Sequence[T]], dict[str, bool]]:
        """
        For each feature in X, return the available unique splits.

        Parameters
        ----------
        force_eval : bool, optional
            If True, force re-evaluation of available splits (default is False).
        assert_cached : bool, optional
            If True, raise an error if a cached value is not available (default is False).

        Returns
        -------
        tuple
            A tuple containing:
              - A dictionary mapping feature names to sorted unique available split values.
              - A dictionary mapping feature names to booleans indicating if the feature is categorical.
        """
        if self.avail_splits is not None and not force_eval:
            return self.avail_splits
        else:
            if assert_cached:
                raise ValueError('No cached value of available splits')
            is_cat = {}
            X = self.X
            avail_vars = {}
            for col_name in X.columns:
                col = X[col_name]
                avail_splits = np.unique(col)
                if len(avail_splits) >= 2:
                    if self._is_categorical(col):
                        avail_vars[col_name] = avail_splits
                        is_cat[col_name] = True
                    else:
                        avail_vars[col_name] = avail_splits[1:]
                        is_cat[col_name] = False
            self.avail_splits = avail_vars, is_cat
        return avail_vars, is_cat

    def reset_avail_splits(self):
        """
        Reset the cached available splits. Necessary whenever the underlying dataset changes
        """
        self.avail_splits = None

    def sample_split(self, split_var: str, avail_vals: Sequence[T]) -> tuple[bool, Sequence[T]|T]:
        if (is_cat:= self._is_categorical(self.X[split_var])):
            split_vals = self._sample_cat_split_subset(avail_vals)
        else:
            split_vals = self.rng.choice(avail_vals)
        return is_cat, split_vals
        

    def _sample_cat_split_subset(self, split_vals: Sequence[T]) -> Sequence[T]:
        """
        Sample a random subset of the available split values for a categorical variable.

        Parameters
        ----------
        split_vals : Sequence[T]
            The available split values.

        Returns
        -------
        Sequence[T]
            A sorted array of sampled split values.
        """
        # enumerating all subsets and then choosing one is too much. So compute the probability of picking
        # a subset of size k, among all possible subsets, then with such probabilities sample
        # a subset size. Finally, sample k values out the available ones, at random.
        
        # make sure you exclude subsets of size n because they don't produce available splits!

        if len(split_vals) ==  1:
            return split_vals
        n = len(split_vals) - 1
        cases = list(map(lambda x: math.comb(n+1, x), range(1, n+1)))
        k = self.rng.choice(np.arange(1, n+1), p=cases/np.sum(cases))
        idx = self.rng.choice(len(split_vals), size=k, replace=False)

        assert len(split_vals[idx]) > 0 and (len(split_vals) - len(split_vals[idx])) > 0

        return np.sort(split_vals[idx]) # type: ignore
    
    def calc_avail_split_and_vars(self) -> tuple[int, int]:
        """
        Compute the number of available variables and the number of available splits for the current split variable.

        Returns
        -------
        tuple
            A tuple (n_avail_vars, n_splits).
        """
        avail_vars, is_cat = self.get_available_splits()
        n_avail_vars = len(avail_vars)
        split_var = self.get_split_var()
        split_vals = avail_vars[split_var]
        if is_cat[split_var]:
            n = len(split_vals) - 1
            cases = list(map(lambda x: math.comb(n+1, x), range(1, n+1)))
            n_splits = np.sum(cases)
        else:
            n_splits = len(split_vals)
        return n_avail_vars, n_splits
        

    def get_split_data(self, split_var: str, split_val: Sequence[T] | T, left_params: Any, right_params: Any) -> tuple[Self, Self]:
        """
        Split the current node's data into two subsets (left and right) based on a given split rule
        and generate new NodeData objects for the children.

        Parameters
        ----------
        split_var : str
            The feature on which to split.
        split_val : Sequence[T] or T
            The split value(s).
        left_params : Any
            Parameters for the left child.
        right_params : Any
            Parameters for the right child.

        Returns
        -------
        tuple
            A tuple (l_node_data, r_node_data) corresponding to the left and right NodeData objects.

        Raises
        ------
        AbstractMethodError
            Always raised; this method should be implemented in a subclass.
        """
        raise AbstractMethodError()
    
    
    def get_data_split(self, split_var: str, split_val: Sequence[T] | T) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into left and right subsets based on the split rule. Returns the left and right data subsets.

        Parameters
        ----------
        split_var : str
            The feature on which to split.
        split_val : Sequence[T] or T
            The split value(s).

        Returns
        -------
        tuple
            A tuple (left_X, right_X, left_y, right_y).
        """
        X = self.X
        y = self.y
        if (self._is_categorical(self.X[split_var])):
            mask = X[split_var].isin(split_val) # type: ignore
        else:
            mask = X[split_var] < split_val
        left_X = X.loc[mask,:]
        right_X = X[~mask]
        left_y = y[mask]
        right_y = y[~mask]
        return left_X, right_X, left_y, right_y


    def update_split_info(self, split_var: str, split_val: Sequence[T] | T):
        self.split_var = split_var
        self.split_set = split_val
        self.is_cat_split = self._is_categorical(self.X[split_var])

    def reset_split_info(self):
        self.split_var = ''
        self.split_set = ''
        self.is_cat_split = False

    def update_split_data(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.reset_avail_splits()

    def is_split_rule_emtpy(self) -> bool:
        return (self.split_var == '') and (self.split_set == '') and (self.is_cat_split == False)


    def get_params(self, print: bool = False) -> Any:
        """
        Retrieve the node-specific parameters (e.g., mu and sigma for regression, or p for classification).

        Parameters
        ----------
        print : bool, optional
            If True, return a formatted string (default is False).

        Returns
        -------
        Any
            The node parameters.
        
        Raises
        ------
        AbstractMethodError
            Always raised; should be implemented in a subclass.
        """
        raise AbstractMethodError()
    
    def count_values(self) -> NDArrayInt:
        """
        Compute the number of observations per class (classification only).

        Returns
        -------
        NDArrayInt
            Array of counts for each class.
        
        Raises
        ------
        AbstractMethodError
            Always raised; should be implemented in a subclass.
        """
        raise AbstractMethodError()
    
    def get_data_averages(self) -> tuple[float, float, float]:
        """
        Compute the number of observations, the mean, and un-normalized variance (regression only).

        Returns
        -------
        tuple
            (n, mean, un-normalized variance)
        
        Raises
        ------
        AbstractMethodError
            Always raised; should be implemented in a subclass.
        """
        raise AbstractMethodError()

    def update_node_params(self, params: tuple[float, float] | NDArrayFloat):
        """
        Update the node parameters with new values.

        Parameters
        ----------
        params : tuple[float, float] or NDArrayFloat
            The new parameter values.
        
        Raises
        ------
        AbstractMethodError
            Always raised; should be implemented in a subclass.
        """
        raise AbstractMethodError()

    def get_preds(self) -> tuple[NDArrayFloat, float]:
        """
        Compute the model predictions (e.g., posterior mean).

        Returns
        -------
        tuple
            A tuple (indices, predictions).
        
        Raises
        ------
        AbstractMethodError
            Always raised; should be implemented in a subclass.
        """
        raise AbstractMethodError()



class NodeDataRegression(NodeData):
    """
    Node data class for regression trees.

    Attributes
    ----------
    mu : float
        The mean parameter.
    sigma : float
        The standard deviation parameter.
    """ 
    # @wraps(NodeData.__init__)
    def __init__(self, mu: float, sigma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.sigma = sigma

    
    def get_mu(self, print: bool = False) -> float:
        return self.mu if not print else self._print(self.mu)
    

    def get_sigma(self, print: bool = False) -> float:
        return self.sigma if not print else self._print(self.sigma)
    
    
    def get_params(self, print: bool = False) -> tuple[float, float]|str:
        if not print:
            return self.mu, self.sigma
        else:
            return f'{self.get_mu(print=True)} {self.get_sigma(print=True)}'
        

    def get_split_data(self, split_var: str, split_val: Sequence[T] | T, left_params: tuple[float,float], right_params: tuple[float,float]):
        """
        Split the data for regression and generate new NodeDataRegression objects for the left and right children.

        Parameters
        ----------
        split_var : str
            The feature to split on.
        split_val : Sequence[T] or T
            The splitting threshold (or set).
        left_params : tuple[float, float]
            Parameters (mu, sigma) for the left child.
        right_params : tuple[float, float]
            Parameters (mu, sigma) for the right child.

        Returns
        -------
        tuple
            (l_node_data, r_node_data)
        """
        left_X, right_X, left_y, right_y = self.get_data_split(split_var, split_val)

        if self.debug:
            assert len(left_params) == 2
            assert len(right_params) == 2
        mu_l, sigma_l = left_params
        mu_r, sigma_r = right_params

        # With this construction it behaves properly with subclasses
        l_node_data = self.__class__(X=left_X, y=left_y, mu=mu_l, sigma=sigma_l, rng=self.rng, debug=self.debug, node_min_size=self.node_min_size)
        r_node_data = self.__class__(X=right_X, y=right_y, mu=mu_r, sigma=sigma_r, rng=self.rng, debug=self.debug, node_min_size=self.node_min_size)
        return l_node_data, r_node_data
    
    def get_data_averages(self) -> tuple[float, float, float]:
        """
        Compute the number of observations, mean, and un-normalized variance for regression.

        Returns
        -------
        tuple
            (n, mean, uvar)
        """
        n = self.get_nobs()
        mean = self.y.mean()
        uvar = (self.y-mean).pow(2).sum() 
        return n, mean, uvar

    def update_node_params(self, params: tuple[float, float] | NDArrayFloat):
        self.mu, self.sigma = params

    def get_preds(self) -> tuple[NDArrayFloat, float]:
        """
        Get predictions for regression.

        Returns
        -------
        tuple
            (indices, predicted constant value equal to mu)
        """
        idx = self.y.index.to_numpy()
        return idx, self.get_mu()


class NodeDataClassification(NodeData):
    """
    Node data class for classification trees.

    Attributes
    ----------
    p : NDArrayFloat
        The probability vector for the classes.
    """

    # @wraps(NodeData.__init__)
    def __init__(self, p: NDArrayFloat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def get_params(self, print: bool = False) -> NDArrayFloat|str:
        if print:
            return ''
        else:
            return self.p
    
    def get_split_data(self, split_var: str, split_val: Sequence[T] | T, left_params: NDArrayFloat, right_params: NDArrayFloat):
        """
        Split the data for classification and generate new NodeDataClassification objects.

        Parameters
        ----------
        split_var : str
            The feature to split on.
        split_val : Sequence[T] or T
            The splitting rule.
        left_params : NDArrayFloat
            New class probability vector for the left child.
        right_params : NDArrayFloat
            New class probability vector for the right child.

        Returns
        -------
        tuple
            (l_node_data, r_node_data)
        """
        left_X, right_X, left_y, right_y = self.get_data_split(split_var, split_val)


        l_node_data = self.__class__(X=left_X, y=left_y, p=left_params, rng=self.rng, debug=self.debug, node_min_size=self.node_min_size)
        r_node_data = self.__class__(X=right_X, y=right_y, p=right_params, rng=self.rng, debug=self.debug, node_min_size=self.node_min_size)
        return l_node_data, r_node_data
    
    def count_values(self) -> NDArrayInt:
        return self.y.value_counts(sort=False).to_numpy()
    
    def update_node_params(self, params: NDArrayFloat):
        self.p = params

    
class NodeDataFast(NodeData):
    """
    Efficient implementation of NodeData, reduces the cost of copy operations.
    """
    def copy(self, light: bool = False, no_data: bool = False, memo: dict|None = None) -> 'NodeData':
        ''' Instead of thoroughly copying the object, just copy what we know we need.'''
        if not light:
            return super().copy(light=light, no_data=no_data, memo=memo)
        else:
            cls = self.__class__
            result = cls.__new__(cls)
            result.node_min_size = self.node_min_size
            result.debug = self.debug
            result.split_var = self.split_var
            result.split_set = self.split_set
            result.is_cat_split = self.is_cat_split
            if not no_data:
                result.rng = self.rng
                result.avail_splits = self.avail_splits
                result._X = self._X
                result._y = self._y
            else:
                result.nobs = self.get_nobs() # type: ignore
                result.avail_splits = None
            return result


class NodeDataRegressionFast(NodeDataFast, NodeDataRegression):
    def copy(self, *args, **kwargs):
        result = super().copy(*args, **kwargs)
        result.mu = self.mu  # type: ignore
        result.sigma = self.sigma # type: ignore
        return result
    
    def get_data_averages(self) -> tuple[float, float, float]:
        """
        Compute the number of observations, mean, and un-normalized variance using fast routines.

        Returns
        -------
        tuple
            (n, mean, uvar)
        """
        y = self.y.to_numpy()
        n = y.shape[0]
        mean = y.mean()
        uvar = ((y-mean)**2).sum() 
        return n, mean, uvar


class NodeDataClassificationFast(NodeDataFast, NodeDataClassification):
    def copy(self, *args, **kwargs):
        result = super(NodeDataFast, self).copy(*args, **kwargs)
        result.p = self.p # type: ignore
        return result

