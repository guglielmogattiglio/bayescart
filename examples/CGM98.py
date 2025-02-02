"""
Reproduce the approxiamte posterior sampling result shown in the CGM98 paper.
"""
from joblib import Parallel, delayed
import numpy as np
from bayescart import BCARTClassic, NodeDataRegression, Tree, sim_cgm98
import pandas as pd
import matplotlib.pyplot as plt


def do(X, y, seed):
    bcart = BCARTClassic(X, y, alpha=0.95, beta=1, a=1/3, mu_bar=4.85, nu=10, lambd=4, iters=1010, burnin=10, thinning=4, move_prob=[1,1,1,1], light=False, seed=seed, verbose='v')
    res = bcart.run()
    return res['posterior_store'], res['integr_llik_store'], res['tree_term_reg']

def unravel(l):
    out = []
    [out.extend(x) for x in l]
    return out

seed = 34647
rng = np.random.default_rng(seed)
X, y = sim_cgm98(800, rng)
res = Parallel(-1)(delayed(lambda i: do(X, y, i))(seed+i) for i in range(10))
post, llik, term_reg = zip(*res)

# True tree
root_node_data = NodeDataRegression(np.nan, np.nan, X, y, rng=rng, debug=True, node_min_size=1)

true_tree = Tree(root_node_data=root_node_data, rng=rng, debug=True, node_min_size=1)
root = true_tree.get_root()
root.update_split_info('v2', ['a','b'])
root.update_node_params((np.nan, np.nan))
node2 = true_tree.add_node(root_node_data.copy(), is_l=True, parent=root)
node2.update_split_info('v1', 5)
node2.update_node_params((np.nan, np.nan))
node3 = true_tree.add_node(root_node_data.copy(), is_l=False, parent=root)
node3.update_split_info('v1', 3)
node3.update_node_params((np.nan, np.nan))
node4 = true_tree.add_node(root_node_data.copy(), is_l=True, parent=node2)
node4.update_node_params((8, 2))
node4.reset_split_info()
node5 = true_tree.add_node(root_node_data.copy(), is_l=False, parent=node2)
node5.update_node_params((2, 2))
node5.reset_split_info()
node6 = true_tree.add_node(root_node_data.copy(), is_l=True, parent=node3)
node6.update_node_params((1, 2))
node6.reset_split_info()
node7 = true_tree.add_node(root_node_data.copy(), is_l=False, parent=node3)
node7.update_split_info('v1', 7)
node7.update_node_params((np.nan, np.nan))
node8 = true_tree.add_node(root_node_data.copy(), is_l=True, parent=node7)
node8.update_node_params((5, 2))
node8.reset_split_info()
node9 = true_tree.add_node(root_node_data.copy(), is_l=False, parent=node7)
node9.update_node_params((8, 2))
node9.reset_split_info()

true_tree.update_subtree_data(root)
true_tree.is_valid()


bcart_truth = BCARTClassic(X, y, alpha=0.95, beta=1, a=1/3, mu_bar=4.85, nu=10, lambd=4, iters=1010, burnin=10, thinning=1, move_prob=[1,1,1,1], debug=True, light=False, seed=rng)
true_llik = bcart_truth.calc_llik(true_tree)
true_post = bcart_truth.get_log_posterior_prob(true_tree)
#%

fig,ax = plt.subplots(1,3, figsize=(9,3)) # (8,3) is also good, maye better)
ax[0].plot(unravel(post))
ax[0].axhline(true_post, color='black')

# fig,ax = plt.subplots()
ax[1].plot(unravel(llik))
ax[1].axhline(true_llik, color='black')

# fig,ax = plt.subplots()
ax[2].plot(unravel(term_reg))
ax[2].axhline(5, color='black')
fig.tight_layout()
