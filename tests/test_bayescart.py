import copy

import numpy as np
import pandas as pd
import pytest

from bayescart import (
    BCARTClassic,
    BCARTGeom,
    BCARTGeomLik,
    BCARTPseudoPrior,
    sim_cgm98
)
from bayescart.tree import Tree
from bayescart.node import Node
from bayescart.node_data import (
    NodeDataRegression,
    NodeDataClassification,
)
from bayescart import eval as bceval
from bayescart import utils

# =============================================================================
# Fixtures for regression and classification data
# =============================================================================
@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 100
    X, y = sim_cgm98(n, rng)
    return X, y, rng

@pytest.fixture
def classification_data():
    rng = np.random.default_rng(42)
    n = 100
    # Create a small DataFrame with one numerical and one categorical feature.
    X = pd.DataFrame({
        'feature_num': rng.normal(size=n),
        'feature_cat': pd.Categorical(rng.choice(['A', 'B', 'C'], size=n))
    })
    # Create a categorical response with two classes.
    y = pd.Series(rng.choice(['class1', 'class2'], size=n), dtype="category")
    return X, y, rng

# =============================================================================
# Tests for sim_cgm98
# =============================================================================
def test_sim_cgm98():
    """Test that sim_cgm98 returns a DataFrame and Series with the expected shape and columns."""
    seed = 42
    rng = np.random.default_rng(seed)
    X, y = sim_cgm98(100, rng)
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, pd.Series), "y should be a Series"
    for col in ['v1', 'v2']:
        assert col in X.columns
    assert X.shape[0] == 100
    assert y.shape[0] == 100

# =============================================================================
# Tests for BCARTClassic (Regression)
# =============================================================================
def test_bcart_classic_run_regression(regression_data):
    """
    Create a BCARTClassic instance for regression using simulated data.
    Run a short chain and verify that key outputs are present.
    """
    seed = 42
    X, y, _ = regression_data
    bcart = BCARTClassic(
        X, y,
        alpha=0.95, beta=1, a=0.3, mu_bar=0.0,
        nu=3, lambd=0.1,
        iters=50, burnin=10, thinning=1,
        move_prob=[1, 1, 1, 1],
        light=True,
        seed=seed
    )
    res = bcart.run()
    for key in ['tree_term_reg', 'integr_llik_store', 'timings', 'setup', 'data']:
        assert key in res, f"Missing key '{key}' in result"
    # Check that the terminal regions are positive numbers.
    assert np.all(res['tree_term_reg'] > 0)

def test_bcart_classic_tree_copy_and_equality(regression_data):
    """
    After performing an update, copy the tree (light copy) and check that the copy
    is equal in structure to the original.
    """
    seed = 42
    X, y, _ = regression_data
    bcart = BCARTClassic(
        X, y,
        alpha=0.95, beta=1, a=0.3, mu_bar=0.0,
        nu=3, lambd=0.1,
        iters=20, burnin=5, thinning=1,
        move_prob=[1, 1, 1, 1],
        light=True,
        seed=seed
    )
    # Force a 'grow' move to update the tree.
    bcart._update_once('grow')
    original_tree = bcart.tree
    tree_copy = original_tree.copy(light=True)
    # Check basic equality (hard=0)
    assert original_tree.is_equal(tree_copy, hard=0), "Tree copy does not match the original structure."

def test_bcart_classic_classification_run():
    """
    Create a BCARTClassic instance for classification using a small categorical dataset.
    Verify that the classification flag is set.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        'v1': [1, 2, 3, 4, 5, 6],
        'v2': ['a', 'b', 'a', 'b', 'a', 'b']
    })
    X['v2'] = X['v2'].astype('category')
    y = pd.Series(['yes', 'no', 'yes', 'no', 'yes', 'no'], dtype="category")
    bcart = BCARTClassic(
        X, y,
        alpha=0.95, beta=1, a=0.3, mu_bar=0.0,
        nu=3, lambd=0.1,
        iters=30, burnin=5, thinning=1,
        move_prob=[1, 1, 1, 1],
        light=True,
        seed=seed
    )
    assert bcart.is_classification, "BCART should be in classification mode for categorical y"
    res = bcart.run()
    assert 'tree_term_reg' in res

# =============================================================================
# Tests for Parallel Tempering Implementations (PT classes)
# =============================================================================
@pytest.mark.parametrize("cls,temps,extra", [
    (BCARTGeom, (1, 0.5), {}),
    (BCARTGeomLik, (1, 0.5), {}),
    (BCARTPseudoPrior, (1, 0.5), {'pprior_alpha': 0.95, 'pprior_beta': 1})
])
def test_parallel_tempering(cls, temps, extra, regression_data):
    """
    Instantiate each parallel tempering variant with a short chain and verify that
    PT-specific keys (such as 'PT_swap_stats') appear in the run result.
    """
    seed = 42
    X, y, _ = regression_data
    bcart_pt = cls(
        X, y,
        alpha=0.95, beta=1, a=0.3, mu_bar=0.0,
        nu=3, lambd=0.1,
        iters=30, burnin=5, thinning=1,
        move_prob=[1, 1, 1, 1],
        light=True,
        seed=seed,
        temps=temps,
        **extra
    )
    res = bcart_pt.run()
    assert 'PT_swap_stats' in res

def test_bcart_task_flag(regression_data, classification_data):
    """
    Check that BCARTClassic sets the is_classification flag correctly based on the dtype of y.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    # Regression case: y is numeric.
    X_reg, y_reg, _ = regression_data
    bcart_reg = BCARTClassic(
        X_reg, y_reg,
        alpha=0.95, beta=1, a=0.3, mu_bar=0.0,
        nu=3, lambd=0.1,
        iters=30, burnin=5, thinning=1,
        move_prob=[1, 1, 1, 1],
        light=True,
        seed=seed
    )
    assert not bcart_reg.is_classification, "Numeric y should trigger regression mode"
    # Classification case:
    X_clf, y_clf, _ = classification_data
    bcart_clf = BCARTClassic(
        X_clf, y_clf,
        alpha=0.95, beta=1, a=0.3, mu_bar=0.0,
        nu=3, lambd=0.1,
        iters=30, burnin=5, thinning=1,
        move_prob=[1, 1, 1, 1],
        light=True,
        seed=seed
    )
    assert bcart_clf.is_classification, "Categorical y should trigger classification mode"

# =============================================================================
# Tests for NodeDataRegression and NodeDataClassification functionality
# =============================================================================
def test_node_data_regression_get_data_averages(regression_data):
    X, y, rng = regression_data
    node_data = NodeDataRegression(mu=0.0, sigma=1.0, X=X, y=y, rng=rng, debug=False, node_min_size=5)
    n_obs, mean, uvar = node_data.get_data_averages()
    assert n_obs == len(y)
    np.testing.assert_almost_equal(mean, y.mean(), err_msg="Mean does not match")
    np.testing.assert_almost_equal(uvar, ((y - y.mean())**2).sum(), err_msg="Unnormalized variance does not match")

def test_node_data_regression_update_and_get_preds(regression_data):
    X, y, rng = regression_data
    node_data = NodeDataRegression(mu=5.0, sigma=2.0, X=X, y=y, rng=rng, debug=False, node_min_size=5)
    node_data.update_node_params((10.0, 3.0))
    params = node_data.get_params()
    assert params == (10.0, 3.0)
    idx, preds = node_data.get_preds()
    np.testing.assert_array_equal(idx, y.index.to_numpy())
    assert preds == 10.0

def test_node_data_classification_count_and_update(classification_data):
    X, y, rng = classification_data
    p = np.array([0.3, 0.7])
    node_data = NodeDataClassification(p=p, X=X, y=y, rng=rng, debug=False, node_min_size=5)
    counts = node_data.count_values()
    assert len(counts) == len(y.cat.categories)
    assert counts.sum() == len(y)
    new_p = np.array([0.2, 0.8])
    node_data.update_node_params(new_p)
    np.testing.assert_array_equal(node_data.get_params(), new_p)

def test_node_data_copy_and_light_copy(regression_data):
    X, y, rng = regression_data
    node_data = NodeDataRegression(mu=3.0, sigma=1.5, X=X, y=y, rng=rng, debug=False, node_min_size=5)
    node_copy = node_data.copy(light=False)
    assert node_copy.mu == node_data.mu and node_copy.sigma == node_data.sigma
    node_data.update_node_params((10.0, 2.0))
    assert node_copy.mu != node_data.mu

# =============================================================================
# Tests for Node and Tree functionality
# =============================================================================
def create_simple_regression_node():
    """Helper: create a Node with NodeDataRegression on a small dataset."""
    X = pd.DataFrame({'v1': [1, 2, 3, 4]})
    y = pd.Series([1, 2, 3, 4])
    node_data = NodeDataRegression(mu=0.0, sigma=1.0, X=X, y=y, rng=np.random.default_rng(42), debug=True, node_min_size=1)
    return Node(id=0, is_l=True, data=node_data, rng=np.random.default_rng(42), debug=True)

def test_node_get_nobs_and_copy():
    node = create_simple_regression_node()
    assert node.get_nobs() == 4
    node_copy = node.copy(light=True)
    assert node is not node_copy
    assert node.get_nobs() == node_copy.get_nobs()

def test_node_update_and_retrieve_split_info():
    node = create_simple_regression_node()
    split_var, split_val, is_cat = node.get_split_info()
    assert split_var == ''
    assert split_val == ''
    assert is_cat is False
    node.update_split_info('v1', 2.5)
    split_var, split_val, is_cat = node.get_split_info()
    assert split_var == 'v1'
    assert split_val == 2.5

def test_node_get_new_split_numeric():
    X = pd.DataFrame({'v1': list(range(1, 11))})
    y = pd.Series(list(range(1, 11)))
    node_data = NodeDataRegression(mu=0.0, sigma=1.0, X=X, y=y, rng=np.random.default_rng(42), debug=True, node_min_size=1)
    node = Node(id=1, is_l=False, data=node_data, rng=np.random.default_rng(42), debug=True)
    split_var, split_val = node.get_new_split()
    assert split_var == 'v1'
    avail_vars, _ = node_data.get_available_splits()
    assert split_val in avail_vars['v1']

def test_node_get_data_split_numeric():
    X = pd.DataFrame({'v1': [1, 2, 3, 4, 5, 6]})
    y = pd.Series([10, 20, 30, 40, 50, 60])
    node_data = NodeDataRegression(mu=0.0, sigma=1.0, X=X, y=y, rng=np.random.default_rng(42), debug=True, node_min_size=1)
    # Using the underlying data splitting method from NodeData
    left_X, right_X, left_y, right_y = node_data.get_data_split('v1', 4)
    if not left_X.empty:
        assert left_X['v1'].max() < 4
    if not right_X.empty:
        assert right_X['v1'].min() >= 4
    assert len(left_X) + len(right_X) == 6

def create_simple_tree():
    """Helper: create a simple Tree with a regression root node."""
    X = pd.DataFrame({'v1': np.arange(1, 11)})
    y = pd.Series(np.arange(1, 11))
    root_data = NodeDataRegression(mu=0.0, sigma=1.0, X=X, y=y, rng=np.random.default_rng(42), debug=True, node_min_size=2)
    return Tree(root_node_data=root_data, rng=np.random.default_rng(42), node_min_size=2, debug=True)

def test_tree_root_properties():
    tree = create_simple_tree()
    root = tree.get_root()
    assert root.id == 0
    assert root.depth == 0
    assert tree.is_stump()

def test_tree_add_node_and_structure():
    tree = create_simple_tree()
    root = tree.get_root()
    child_data_left = copy.deepcopy(root._data)
    child_data_right = copy.deepcopy(root._data)
    left_child = tree.add_node(data=child_data_left, is_l=True, parent=root)
    right_child = tree.add_node(data=child_data_right, is_l=False, parent=root)
    assert len(tree.nodes) == 3
    assert left_child.depth == root.depth + 1
    assert right_child.depth == root.depth + 1
    children = tree.get_children(root)
    assert len(children) == 2
    assert tree.get_parent(left_child) == root
    assert tree.get_sibling(left_child) == right_child

def test_tree_sample_leaf():
    tree = create_simple_tree()
    root = tree.get_root()
    tree.apply_split(root, split_var='v1', split_val=5, l_leaf_params=(1.0, 1.0), r_leaf_params=(2.0, 1.0))
    leaf = tree.sample_leaf(node_min_size=2)
    assert leaf.is_leaf()
    assert leaf.get_nobs() >= 2

def test_tree_apply_split_and_validity():
    tree = create_simple_tree()
    root = tree.get_root()
    tree.apply_split(root, split_var='v1', split_val=6, l_leaf_params=(1.0, 1.0), r_leaf_params=(2.0, 1.0))
    assert not tree.is_stump()
    assert tree.is_valid()
    assert tree.get_n_leaves() == 2

def test_tree_remove_node():
    tree = create_simple_tree()
    root = tree.get_root()
    child_data_left = copy.deepcopy(root._data)
    child_data_right = copy.deepcopy(root._data)
    left_child = tree.add_node(data=child_data_left, is_l=True, parent=root)
    right_child = tree.add_node(data=child_data_right, is_l=False, parent=root)
    removed = tree.remove_node(left_child)
    assert removed == 1
    assert left_child.id not in tree.nodes

def test_tree_update_subtree_data():
    tree = create_simple_tree()
    root = tree.get_root()
    tree.apply_split(root, split_var='v1', split_val=5, l_leaf_params=(1.0, 1.0), r_leaf_params=(2.0, 1.0))
    root.update_split_info('v1', 7)
    tree.update_subtree_data(root)
    left_X, right_X, _, _ = root._data.get_data_split('v1', 7)
    if not left_X.empty:
        assert left_X['v1'].max() < 7
    if not right_X.empty:
        assert right_X['v1'].min() >= 7

def test_tree_is_equal():
    tree1 = create_simple_tree()
    tree2 = create_simple_tree()
    assert tree1.is_equal(tree2, hard=0)
    tree1.apply_split(tree1.get_root(), split_var='v1', split_val=5, l_leaf_params=(1.0, 1.0), r_leaf_params=(2.0, 1.0))
    assert not tree1.is_equal(tree2, hard=0)
    tree2.apply_split(tree2.get_root(), split_var='v1', split_val=5, l_leaf_params=(1.0, 1.0), r_leaf_params=(2.0, 1.0))
    assert tree1.is_equal(tree2, hard=1)

def test_tree_fast_copy():
    tree = create_simple_tree()
    tree.apply_split(tree.get_root(), split_var='v1', split_val=5, l_leaf_params=(1.0, 1.0), r_leaf_params=(2.0, 1.0))
    tree_fast = tree.copy(light=True)
    assert tree_fast.is_valid()

# =============================================================================
# Tests for Evaluation Module Functions
# =============================================================================
def test_eval_calc_log_tree_prob_and_llik():
    rng = np.random.default_rng(1201)
    X, y = sim_cgm98(60, rng)
    model = BCARTClassic(X, y, iters=40, burnin=10, thinning=2,
                          seed=1201, verbose='v', light=True)
    result = model.run()
    tree = next((t for t in result['tree_store'] if t is not None), None)
    assert tree is not None
    prior, llik = bceval.calc_log_tree_prob_and_llik(tree, result)
    assert isinstance(prior, float)
    assert isinstance(llik, float)

def test_produce_tree_table(regression_data):
    X, y, _ = regression_data
    runs = []
    for seed_val in range(40, 45):
        model = BCARTClassic(
            X, y,
            alpha=0.95, beta=1, a=0.3, mu_bar=0.0,
            nu=3, lambd=0.1,
            iters=30, burnin=5, thinning=1,
            move_prob=[1, 1, 1, 1],
            light=True,
            seed=seed_val, verbose='v'
        )
        runs.append(model.run())
    table = bceval.produce_tree_table(runs)
    assert isinstance(table, pd.DataFrame)
    for col in ['Mn', 'Std', 'b']:
        assert col in table.columns

# =============================================================================
# Additional Integration Tests
# =============================================================================
def test_integration_multiple_models_different_params():
    rng = np.random.default_rng(1501)
    X, y = sim_cgm98(80, rng)
    results = []
    for s in [1501, 1502, 1503, 1504]:
        model = BCARTClassic(
            X, y,
            iters=50, burnin=10, thinning=2,
            seed=s, verbose='v', light=True, move_prob=[1,1,1,1]
        )
        results.append(model.run())
    for res in results:
        for key in ['tree_store', 'integr_llik_store', 'tree_term_reg']:
            assert key in res

def test_integration_tempering_odd_chains_error():
    rng = np.random.default_rng(1601)
    X, y = sim_cgm98(50, rng)
    temps = (1, 0.8, 0.6)  # odd number of temperatures
    with pytest.raises(ValueError):
        BCARTGeom(X, y, iters=30, burnin=10, thinning=2,
                  seed=1601, verbose='v', light=True, temps=temps)

def test_integration_swap_prob_over_time_exists():
    rng = np.random.default_rng(1801)
    X, y = sim_cgm98(50, rng)
    temps = (1, 0.8, 0.6, 0.4)
    for Model in [BCARTGeom, BCARTGeomLik, BCARTPseudoPrior]:
        kwargs = dict(iters=50, burnin=5, thinning=1, seed=1801, verbose='v', light=True, temps=temps)
        if Model is BCARTPseudoPrior:
            kwargs.update({'pprior_alpha': 0.95, 'pprior_beta': 1.5})
        model = Model(X, y, **kwargs)
        result = model.run()
        stats = result.get('PT_swap_stats', {})
        assert isinstance(stats.get('PT_swap_prob_over_time', {}), dict)

def test_integration_tree_validity_after_updates():
    rng = np.random.default_rng(1901)
    X, y = sim_cgm98(70, rng)
    model = BCARTClassic(X, y, iters=50, burnin=10, thinning=2,
                          seed=1901, verbose='v', light=True)
    result = model.run()
    final_tree = model.tree
    assert final_tree.is_valid()

def test_integration_cross_model_consistency():
    rng = np.random.default_rng(2101)
    X, y = sim_cgm98(50, rng)
    model_classes = [BCARTClassic, BCARTGeom, BCARTGeomLik, BCARTPseudoPrior]
    for Model in model_classes:
        kwargs = dict(iters=30, burnin=5, thinning=1, seed=2101, verbose='v', light=True)
        if Model in [BCARTGeom, BCARTGeomLik, BCARTPseudoPrior]:
            kwargs['temps'] = (1, 0.8, 0.6, 0.4)
            if Model is BCARTPseudoPrior:
                kwargs.update({'pprior_alpha': 0.95, 'pprior_beta': 1.5})
        model = Model(X, y, **kwargs)
        result = model.run()
        tree = model.tree  # cold chain tree
        post_prob = model.get_log_posterior_prob(tree)
        assert np.isfinite(post_prob)

# =============================================================================
# Tests for Plotting Utilities (run without error)
# =============================================================================

def test_plot_chain_comm_and_swap_prob(regression_data):
    X, y, rng = regression_data
    temps = (1, 0.8, 0.6, 0.4)
    model = BCARTGeom(
        X, y,
        iters=30, burnin=5, thinning=1,
        seed=42, verbose='v', light=True, temps=temps
    )
    res = model.run()
    utils.plot_chain_comm(res, title='Chain Communication')
    utils.plot_swap_prob(res, title='Swap Probabilities')
    
# =============================================================================
# Run tests if executed as a script
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__])
