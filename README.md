<div align="center">
  <a href="https://guglielmogattiglio.github.io/bayescart/">
    <img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Documentation Status">
  </a>
  <a href="https://github.com/guglielmogattiglio/bayescart/actions/workflows/docs.yml">
    <img src="https://github.com/guglielmogattiglio/bayescart/actions/workflows/docs.yml/badge.svg" alt="Docs Build">
  </a>
  <a href="https://github.com/guglielmogattiglio/bayescart/actions/workflows/tests.yml">
    <img src="https://github.com/guglielmogattiglio/bayescart/actions/workflows/tests.yml/badge.svg" alt="Tests Status">
  </a>
</div>


# BayesCART

**BayesCART** is a Python package for Bayesian Classification and Regression Trees (CART) posterior sampling using custom, advanced tempering methods.

This package provides classes and functions to build, sample, and evaluate Bayesian
classification and regression trees using Markov chain Monte Carlo (MCMC) methods.
It supports various tempering strategies (geometric, likelihood-based, and pseudo-prior)
to improve mixing in multi-modal posterior distributions.

For theoretical background on Bayesian CART, and the specific tempering strategies implemented in this package, check this [detailed blog series](https://guglielmogattiglio.com/blog/bayesian-classification-and-regression-trees-theoretical-series).

## Features

- **Bayesian CART models:** Build regression and classification trees with a Bayesian framework.
- **Tempering methods:** Improve posterior exploration via parallel tempering.
- **Modular design:** Separate modules for node data, tree structure, and BCART sampling.
- **Test suite:** Automated tests are included.
- **Documentation:** Fully generated via Sphinx and published on GitHub Pages.


For more information on the creation of the package, see [this dedicated page](https://guglielmogattiglio.com/blog/bayescart-python-package/).






## Installation

To install from source

```bash
git clone https://github.com/guglielmogattiglio/bayescart.git
cd bayescart
pip install -e .
```

## Minimal Usage Example

For an more detailed example on how to use this package, see this [tutorial notebook](https://guglielmogattiglio.com/blog/using-bayescart-to-solve-cgm98).

```python
from bayescart import BCARTClassic, sim_cgm98
import numpy as np

# Create example data according to CGM98 example
rng = np.random.default_rng(45356)
X, y = sim_cgm98(800, rng)

# Initialize the model (here using the standard CGM98 BCART sampler)
model = BCARTClassic(X, y, iters=1010, burnin=10, thinning=1)

# Run the sampler
result = model.run()

# Print the result
result["tree_store"][-1].show()
```

## Documentation
The full documentation is available at: https://guglielmogattiglio.github.io/bayescart/

To refresh the documentation, run 
```bash
cd docs
make clean
make html
```

If new modules are added, the `.rst` files need to be updated first. From the project folder, run
```bash
sphinx-apidoc -o docs/source ../bayescart
```

## Citation

If you use this work, please cite:

> Gattiglio, Guglielmo. *Tempered Stochastic Search of Bayesian CART Models.* Milano: Università Bocconi, 2021. 

## License
This project is licensed under the Apache License 2.0.
