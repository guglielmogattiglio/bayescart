"""Tree class for bayescart.

This module defines the Tree class that extends treelib's Tree to support operations
needed for Bayesian CART such as applying splits, copying subtrees, and computing
prior probabilities.
"""

import numpy as np
from treelib import Tree as TreelibTree
from typing import Sequence, Self, Any
from copy import deepcopy
from .node import Node
from .node_data import NodeData, NodeDataClassification
from .mytyping import T
from .exceptions import InvalidTreeError
from .utils import my_choice
from .node import NodeFast

class Tree(TreelibTree):
    """
    Extended tree class for Bayesian CART that builds on the treelib Tree.

    This class maintains a counter for node IDs and provides additional methods for 
    sampling leaves, copying trees, and applying splits.
    
    Attributes
    ----------
    id_counter : int
        Counter for unique node identifiers.
    rng : np.random.Generator
        Random generator used for sampling.
    node_min_size : int
        Minimum number of observations per node.
    debug : bool
        If True, enables additional assertions.
    llik : float or None
        Cached integrated log-likelihood of the tree.
    log_tree_prior_prob : float or None
        Cached log prior probability of the tree.
    temperature : float
        The current temperature for tempering.
    """
    node_class = Node
    def __init__(self, root_node_data: NodeData, rng: np.random.Generator,
                 node_min_size: int, debug: bool):
        super().__init__(node_class=self.node_class)
        self.id_counter: int = 0
        self.rng = rng
        self.node_min_size = node_min_size
        self.debug = debug
        self.llik: float|None = None
        self.log_tree_prior_prob: float|None = None
        self.temperature = np.nan

        self.add_node(root_node_data, is_l=False)
           


    def add_node(self, data: NodeData, is_l: bool, parent: Node | None = None) -> Node:
        node = self.node_class(self.id_counter, is_l=is_l, data=data, rng=self.rng, debug=self.debug)
        node.depth = 0 if parent is None else parent.depth + 1
        super().add_node(node, parent)
        self.id_counter += 1
        return node


    def sample_leaf(self, node_min_size: int) -> Node:
        """
        Sample a leaf node at random from the tree that has at least node_min_size observations.

        Parameters
        ----------
        node_min_size : int
            Minimum required observations.

        Returns
        -------
        Node
            A randomly chosen leaf node.

        Raises
        ------
        InvalidTreeError
            If no leaf with sufficient observations is found.
        """
        leaves = self.get_leaves()
        leaves_sizes = np.array([node.get_nobs() for node in leaves], dtype=np.int_)

        p = np.where(leaves_sizes >= node_min_size, 1, 0)
        if p.sum() == 0:
            raise InvalidTreeError('No valid leaf to split due to min node size constraint')
        else:
            res = my_choice(self.rng, leaves, p=p/np.sum(p))
            return res
        

    def get_leaves(self) -> list[Node]:
        """
        Return all the leaf nodes of the tree.

        Returns
        -------
        list
            A list of leaf nodes.
        """
        return self.leaves()
    
    def get_n_leaves(self) -> int:
        """
        Get the number of leaf nodes.

        Returns
        -------
        int
            The count of leaves.
        """
        return len(self.get_leaves())

    def __deepcopy__(self, memo):
        return self.copy(light=False, memo=memo)

    def copy(self, light: bool = False, no_data: bool = False, memo: dict|None = None) -> 'Tree':
        '''Deep copy the tree with all node info. If light, don't deep-copy (X,y) at each node.'''
        if memo is None:
            memo = {}
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k == '_nodes':
                _nodes = {}
                for nid in self._nodes:
                    _nodes[nid] = self._nodes[nid].copy(light=light, no_data=no_data, memo=memo)
                setattr(result, k, _nodes)
            elif k == 'rng':
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


    def is_valid(self) -> bool:
        """
        Check whether the tree is valid.

        The function recursively checks each node for logical consistency of splits,
        data assignments, and node properties.

        Returns
        -------
        bool
            True if the tree passes all validity checks.
        """
        def is_valid_node(node: Node) -> bool:
            try:
                return _is_valid_node(node)
            except Exception as e:
                node._gen_tags()
                print(f'Error in node {node.tag}')
                raise Exception(f'Error in node {node.tag}, {type(e).__name__}: {e}') from e

        def _is_valid_node(node: Node) -> bool:
            children = self.get_children(node)
            if not node.is_leaf():
                # if not leaf must have two children
                assert (len(children) == 2) 

                # first child is left, second is right
                l_child, r_child = children
                assert l_child.is_l and not r_child.is_l

                # check that we have the right split
                split_var, split_val, is_cat = node.get_split_info()
                left_X, right_X, left_y, right_y = node._data.get_data_split(split_var, split_val)
                assert left_X.equals(l_child._data.X) and right_X.equals(r_child._data.X)
                assert left_y.equals(l_child._data.y) and right_y.equals(r_child._data.y)
            else:
                assert (len(children) == 0)
                split_var, split_val, is_cat = node.get_split_info()
                assert split_var == ''
                assert split_val == ''
                assert is_cat == False

            # if classification task, check K classes and that p sums to one
            if isinstance(node._data, NodeDataClassification):
                p = node._data.p
                assert len(p) == node._data.y.cat.categories.size
                assert np.isclose(p.sum(), 1)

            # node observations must be > min
            assert node.get_nobs() > self.node_min_size

            # check that the node tag descriptions are properly updated every time
            # prev_tag = node.tag
            # node._gen_tags()
            # assert node.tag == prev_tag

            # These check apply only for non-root nodes
            if node.is_root():
                assert node.depth == 0
            else:
                parent = self.get_parent(node)
                if parent is None:
                    raise ValueError('Node parent should not be None.')
                # node obs must be smaller than parent
                assert node.get_nobs() < parent.get_nobs()

                # node depth is 1 + parent depth
                assert node.depth == 1 + parent.depth

                # node depth should also match level implementation
                assert self.level(node.id) == node.depth

            return True

        res = list(filter(is_valid_node, self.all_nodes_itr()))
        return len(self.nodes) == len(res)
    

    def apply_split(self, node: Node, split_var: str, split_val: Sequence[T] | T, l_leaf_params: Any, r_leaf_params: Any):
        """
        Apply a split to a leaf node by generating two children nodes.

        Parameters
        ----------
        node : Node
            The leaf node to be split.
        split_var : str
            The feature to split on.
        split_val : Sequence[T] or T
            The split value(s).
        l_leaf_params : Any
            Parameters for the left child.
        r_leaf_params : Any
            Parameters for the right child.

        Raises
        ------
        InvalidTreeError
            If any of the resulting children have fewer observations than the minimum.
        """
        # Split the node (leaf) by generating two children. Update the splitting rule (previously blank) of the node accordingly. The children parameters are given.

        if self.debug:
            assert node.is_leaf()
            # assert node.is_split_rule_empty()

        # find left and right data subsets
        node_data_l, node_data_r = node.get_split_data(split_var, split_val, l_leaf_params, r_leaf_params)
        if node_data_l.get_nobs() < self.node_min_size or node_data_r.get_nobs() < self.node_min_size:
            raise InvalidTreeError('One of the children has less than min node size observations')
        node.update_split_info(split_var, split_val)

        # Append the nodes and grow the tree
        self.add_node(data=node_data_l, is_l=True, parent=node)
        self.add_node(data=node_data_r, is_l=False, parent=node)


    def get_node(self, node_id: int) -> Node:
        if (node := super().get_node(node_id)) is None:
            raise ValueError(f'Node {node_id} does not exist')
        return node
    
    def get_root(self) -> Node:
        """
        Return the root node of the tree.

        Returns
        -------
        Node
            The root node.
        """
        root = self.get_node(0)
        if self.debug:
            if not root.is_root():
                raise ValueError('Root node is not actually root')
        return root

    def is_stump(self) -> bool:
        # Different implementations of the same thing. Just for sanity
        c1 = self.get_root().is_leaf()
        c3 = self.size() == 1
        c4 = len(self.nodes) == 1
        if self.debug:
            assert c1 == c3 == c4
        return c1
    
    def get_children(self, node: Node|int) -> list[Node]:
        """
        Get the children of the specified node.

        Parameters
        ----------
        node : Node or int
            The node or its identifier.

        Returns
        -------
        list
            A list of child nodes (ordered as left then right).

        Raises
        ------
        ValueError
            If the node does not have exactly two children (in non-leaf cases).
        """
        if isinstance(node, Node):
            res =  self.children(node.id)
        else:
            res = self.children(node)
        if len(res) == 0:
            return []
        if self.debug:
            assert len(res) == 2
        if not res[0].is_l:
            res[0], res[1] = res[1], res[0]

        if self.debug:
            assert res[0].is_l and not res[1].is_l
        return res
    
    def remove_node(self, node: Node|int) -> int:
        if isinstance(node, Node):
            return super().remove_node(node.id)
        else:
            return super().remove_node(node)

    def get_parent(self, node: Node) -> Node:
        if isinstance(node, Node):
            res =  self.parent(node.id)
        else:
            res = self.parent(node)
        if res is None:
            raise ValueError('Node has no parent')
        return res
    
    def get_sibling(self, node: Node) -> Node:
        if isinstance(node, Node):
            res =  self.siblings(node.id)
        else:
            res = self.siblings(node)
        if len(res) != 1:
            raise ValueError(f'Wrong number of siblings for Node {str(node)}: {len(res)}')
        return res[0]
    
    def get_parents_with_two_leaves(self) -> list[Node]:
        """
        Get all internal nodes that have exactly two leaves as children.

        Returns
        -------
        list
            List of nodes satisfying the condition.
        """
        def filter_f(node: Node) -> bool:
            children = self.get_children(node)
            if len(children) == 0:
                return False
            return children[0].is_leaf() and children[1].is_leaf()
        
        return list(self.filter_nodes(filter_f))
    
    def get_nonleaf_nodes(self, filter_root: bool = False) -> list[Node]:
        def filter_f(node: Node) -> bool:
            if node.is_root():
                if filter_root:
                    return False
            return not node.is_leaf()
        return list(self.filter_nodes(filter_f))
    
    def check_split_struct(self, node: Node):
        """
        Check that the splitting rules along the subtree starting at a given node
        do not produce empty splits.

        While this function does not guarantee that the split is valid, its fast 
        to execute and can filter some obvious incompatibilities.

        Parameters
        ----------
        node : Node
            The root of the subtree to check.

        Raises
        ------
        InvalidTreeError
            If an empty split is encountered.
        """
        #Check that the split sequence along a subtree does not lead to (a priori) empty nodes, for example by a v1 => 5 and v1 < 3. This can happen due to swap and change.'''
        def _check_struct_rec(node: Node, d: dict[str, tuple[float,float]|set[Any]],
                              global_d: dict[str, tuple[float,float]|set[Any]]):
            '''Dict contains the exclusion logic.
            if not categorical, [min, max) range of the data. Initialized to (-inf, inf).
            If categorical, the set of possible values. Initialized to the available ones.'''
            if node.is_leaf():
                return
            split_var, split_val, is_cat = node.get_split_info()

            if split_var not in d:
                if split_var not in global_d:
                    raise InvalidTreeError('Split variable is not available')
                d[split_var] = global_d[split_var]

            val = d[split_var]
            if is_cat:
                split_val = set(split_val)
                l_split = val.intersection(split_val) # type: ignore
                if len(l_split) == 0:
                    raise InvalidTreeError('Empty split')
                r_split = val.difference(split_val) # type: ignore
                if len(r_split) == 0:
                    raise InvalidTreeError('Empty split')
            else:
                l_split = (val[0], min(val[1], split_val)) # type: ignore
                if l_split[0] >= l_split[1]:
                    raise InvalidTreeError('Empty split')
                r_split = (max(val[0], split_val), val[1]) # type: ignore
                if r_split[0] >= r_split[1]:
                    raise InvalidTreeError('Empty split')
            # if survived, repeat in both children
            d_l = {k: v for k, v in d.items()}
            d_l[split_var] = l_split
            d_r = {k: v for k, v in d.items()}
            d_r[split_var] = r_split

            l_child, r_child = self.get_children(node)
            _check_struct_rec(l_child, d_l, global_d)
            _check_struct_rec(r_child, d_r, global_d)

        aval_splits, is_cat = node.get_available_splits()
        global_d = {k: set(v) if is_cat[k] else (-np.inf, v[-1]+1) for k, v in aval_splits.items()}
        
        _check_struct_rec(node, {}, global_d)
                
                    
                

    def update_subtree_data(self, node: Node):
        """
        Update the data splits for all descendants of a given node.
        
        Parameters
        ----------
        node : Node
            The node whose subtree data will be updated.
        """
        #If the splitting rules have not been changed from the outside, this should have no effect.
        def _update_split_rec(node: Node):
            '''Compute left and right children data subsets. If valid, update 
            the children subsets. Repeat in children'''

            if node.is_leaf():
                return
            
            # split the data based on the current rule
            left_X, right_X, left_y, right_y = node.get_data_split()

            # attempt to update the data
            l_child, r_child = self.get_children(node)
            l_child.update_split_data(left_X, left_y)
            r_child.update_split_data(right_X, right_y)

            # repeat in children
            _update_split_rec(l_child)
            _update_split_rec(r_child)

        _update_split_rec(node)

    def update_split(self, node: Node, split_var: str, split_val: Sequence[T] | T):
        """
        Change the splitting rule for the current node. Update the data subset of the children, and of all the descendants recursively.

        Parameters
        ----------
        node : Node
            The node to update.
        split_var : str
            The new splitting variable.
        split_val : Sequence[T] or T
            The new splitting value(s).
        """
        # change the split rule for the root node
        node.update_split_info(split_var, split_val)
        self.update_subtree_data(node)

    def show(self):
        """
        Print the tree. Update all the tags first. Tags are not automatically updated for performance speed.
        
        Returns
        -------
        str
            The string representation of the tree.
        """
        for node in self.all_nodes_itr():
            node._gen_tags()
        res = str(self)
        print(res)
        return res
    
    def is_equal(self, other: Self, hard:int = 0, categories: dict[str,set]|None = None) -> bool:
        """
        Check if two trees are equal in structure and (optionally) in parameters.
        
        The hard parameter controls how strict the testing should be. Hard = 0 checks 
        they have the same structure and same splitting variables. Hard = 1 additionally 
        checks the splitting values. Hard = 2 expects same structure, same splits, 
        same data in nodes. Hard = 3 expects equality also for leaf parameters.

        Parameters
        ----------
        other : Tree
            The tree to compare with.
        hard : int, optional
            Level of strictness (0: structure and split variable; 1: includes split values; 2: includes data; 3: also leaf parameters) (default 0).
        categories : dict[str, set] or None, optional
            Mapping of categorical variable values, if needed.

        Returns
        -------
        bool
            True if the trees are equal under the chosen criteria.
        """        
        def _check_nodes(node1: Node, node2: Node) -> bool:

            # same type of node, one must be subclass of the other
            assert isinstance(node1, Node)
            assert isinstance(node2, Node)

            # check right amount of children
            c1 = self.get_children(node1)
            c2 = other.get_children(node2)
            assert len(c1) == len(c2)
            
            # if leaf, check data
            if node1.is_leaf():
                assert node2.is_leaf()

                # check data
                if hard >= 2:
                    assert node1._data.X.equals(node2._data.X)
                    assert node1._data.y.equals(node2._data.y)

                # check leaf params
                if hard >= 3:
                    assert np.allclose(np.array(node1._data.get_params()), np.array(node2._data.get_params()))


            # if interior, check splitting rule
            else:
                assert not node2.is_leaf()
                svar1, sval1, is_cat1 = node1.get_split_info()
                svar2, sval2, is_cat2 = node2.get_split_info()
                if hard >= 0:
                    assert svar1 == svar2
                    assert is_cat1 == is_cat2
                    # assert node1._data.get_split_var() == node2._data.get_split_var()
                if is_cat1:
                    # same split vals
                    cond1 = set(sval1) == set(sval2)

                    # mirrored split vals. E.g. (ab) vs (cd). Children need to be swapped
                    if categories is None:
                        cond2 = set(node1._data.X[svar1].cat.categories).difference(sval1) == set(sval2)
                    else:
                        cond2 = categories[svar1].difference(sval1) == set(sval2)

                    if cond2:
                        # need to swap children of node 1
                        c1 = c1[::-1]
                    if hard >= 1:
                        assert cond1 or cond2
                else:
                    if hard >= 1:
                        assert type(sval1) == type(sval2)
                        if isinstance(sval1, np.ndarray):
                            assert set(sval1) == set(sval2)
                        else:
                            assert np.isclose(sval1, sval2)
            if len(c1) == 0:
                return True
            else:
                return all(map(_check_nodes, c1, c2))
        
        if len(self.nodes) != len(other.nodes):
            return False
        try:
            return _check_nodes(self.get_root(), other.get_root())
        except AssertionError:
            return False

class TreeFast(Tree):
    """
    Fast implementation of Tree using NodeFast as the underlying node class. Tree copy is also improved.
    """
    node_class = NodeFast

    def copy(self, light: bool = False, no_data: bool = False, memo: dict|None = None) -> 'Tree':
        """
        Optimized copy of the tree.
        """
        if not light:
            return super().copy(light=light, no_data=no_data, memo=memo)
        else:
            cls = self.__class__
            result = cls.__new__(cls)
            result._nodes = {k: v.copy(light=light, no_data=no_data) for k,v in self._nodes.items()}
            result.rng = self.rng
            result.node_min_size = self.node_min_size
            result.debug = self.debug
            result.llik = self.llik
            result.log_tree_prior_prob = self.log_tree_prior_prob
            result.id_counter = self.id_counter
            result.ROOT = self.ROOT
            result.DEPTH = self.DEPTH
            result.WIDTH = self.WIDTH
            result.ZIGZAG = self.ZIGZAG
            result.node_class = self.node_class
            result._identifier = self._identifier
            result.root = self.root
            result.temperature = self.temperature

            return result
        
    def update_subtree_data(self, node: Node):
        """
        Update the subtree data with a fast preliminary check before performing the full update.

        The preliminary check does not count how many observations fall in each node; just whether the splits make logical sense. This is fast. If so, proceed with the proper, expensive check.
        """        
        self.check_split_struct(node)
        super().update_subtree_data(node)

    