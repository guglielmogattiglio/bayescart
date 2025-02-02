# bayescart

**bayescart** is a Python package for Bayesian Classification and Regression Trees (CART) posterior sampling using custom, advanced tempering methods.

This package provides classes and functions to build, sample, and evaluate Bayesian
classification and regression trees using Markov chain Monte Carlo (MCMC) methods.
It supports various tempering strategies (geometric, likelihood-based, and pseudo-prior)
to improve mixing in multi-modal posterior distributions.

## Features

- **Bayesian CART models:** Build regression and classification trees with a Bayesian framework.
- **Tempering methods:** Improve posterior exploration via parallel tempering.
- **Modular design:** Separate modules for node data, tree structure, and BCART sampling.
- **Test suite:** Automated tests are included.
- **Documentation:** Fully generated via Sphinx and published on GitHub Pages.

## Installation

To install from source

```bash
git clone https://github.com/guglielmogattiglio/bayescart.git
cd bayescart
pip install -e .
```

## Usage Example

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

## License
This project is licensed under the Apache License 2.0.
