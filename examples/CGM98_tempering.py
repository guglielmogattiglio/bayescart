"""
Demonstration of the capabilities of the posterior sampling algorithms
implemented in this package. Can take several hours to run depending on
your hardware.
"""


from joblib import Parallel, delayed
import numpy as np
from bayescart import BCARTClassic, sim_cgm98, BCARTGeomLik, BCARTGeom, BCARTPseudoPrior
from bayescart.eval import produce_tree_table
from bayescart.utils import plot_hists, plot_chain_comm, plot_swap_prob


base_iter = 50000
base_burnin = 1000
thinning_base = 20
tree_spacing_base = 20

# If the chains mix slowly, then you'll want the spacing to be big enough. 20 is a reasonable guess for this model. Otherwise, the better approach is to dedicate as much compute time as possible. You can always chop the results and/or run again for less.

def do_classic(X, y, seed):
    mult = 8
    bcart = BCARTClassic(X, y, alpha=0.95, beta=1, a=1/3, mu_bar=4.85, nu=10, lambd=4, iters=mult*base_iter, burnin=mult*base_burnin, thinning=mult*thinning_base, move_prob=[1,1,1,1], light=False, seed=seed, verbose='v', store_tree_spacing=mult*tree_spacing_base)
    return bcart.run()

def do_geom(X, y, seed):
    mult = 1
    bcart = BCARTGeom(X, y, alpha=0.95, beta=1, a=1/3, mu_bar=4.85, nu=10, lambd=4, iters=mult*base_iter, burnin=mult*base_burnin, thinning=mult*thinning_base, move_prob=[1,1,1,1], light=False, seed=seed, temps = (1,0.85, 0.7,0.48,0.31,0.2,0.08,1e-7), verbose='v', store_tree_spacing=mult*tree_spacing_base)
    return bcart.run()

def do_geom_lik(X, y, seed):
    mult =  2
    bcart = BCARTGeomLik(X, y, alpha=0.95, beta=1, a=1/3, mu_bar=4.85, nu=10, lambd=4, iters=mult*base_iter, burnin=mult*base_burnin, thinning=mult*thinning_base, move_prob=[1,1,1,1], light=False, seed=seed, temps = (1, 0.04, 0.011, 0.005), verbose='v', store_tree_spacing=mult*tree_spacing_base)
    return bcart.run()

def do_pp(X, y, seed):
    mult= 2 
    bcart = BCARTPseudoPrior(X, y, alpha=0.95, beta=1, a=1/3, mu_bar=4.85, nu=10, lambd=4, iters=mult*base_iter, burnin=mult*base_burnin, thinning=mult*thinning_base, move_prob=[1,1,1,1], light=False, seed=seed, temps = (1, 0.065, 0.028, 0.015), verbose='v', store_tree_spacing=mult*tree_spacing_base, pprior_alpha=0.95, pprior_beta=1.6)
    return bcart.run()

seed = 34647
rng = np.random.default_rng(seed)
X, y = sim_cgm98(800, rng)

# Runtime is about 20mins for each, except Geom whic his 30 min.

res_classic = list(Parallel(-1)(delayed(lambda i: do_classic(X, y, i))(seed+i) for i in range(8)))
res_geom = list(Parallel(-1)(delayed(lambda i: do_geom(X, y, i))(seed+i) for i in range(8)))
res_geom_lik = list(Parallel(-1)(delayed(lambda i: do_geom_lik(X, y, i))(seed+i) for i in range(8)))
res_pp = list(Parallel(-1)(delayed(lambda i: do_pp(X, y, i))(seed+i) for i in range(8)))

mdls = [res_classic, res_geom, res_geom_lik, res_pp]
summary_tbls = [produce_tree_table(res) for res in mdls]

# Comment: clear convergence failures for both classic and geom. GeomLik is closer to converge and properly identifies the top 2 trees. PP is much better, properly identifying all top 5 trees.
for idx, res in enumerate(mdls):
    idx_range = (0,1)
    plot_hists(res, idx_range=idx_range)
    print(summary_tbls[idx])

#%%
idx_range = (0,0.1)
res = res_classic
plot_hists(res, idx_range=idx_range)
# produce_tree_table(res)

#%%


plot_hists(res_classic, idx_range=idx_range)
plot_hists(res_geom, idx_range=idx_range)
plot_hists(res_geom_lik, idx_range=idx_range)
plot_hists(res_pp, idx_range=idx_range)

plot_chain_comm(res_geom[0]) 
plot_swap_prob(res_geom[0])

plot_chain_comm(res_geom_lik[0])
plot_swap_prob(res_geom_lik[0])

plot_chain_comm(res_pp[6])
plot_swap_prob(res_pp[6])