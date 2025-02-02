.. bayescart documentation master file, created by
   sphinx-quickstart on Sat Feb  1 23:26:59 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bayescart documentation
=======================

**bayescart** is a Python package for Bayesian Classification and Regression Trees (CART) posterior sampling using custom, advanced tempering methods.

This package provides classes and functions to build, sample, and evaluate Bayesian
classification and regression trees using Markov chain Monte Carlo (MCMC) methods.
It supports various tempering strategies (geometric, likelihood-based, and pseudo-prior)
to improve mixing in multi-modal posterior distributions.

For more information on the creation of the package, see `this dedicated page <https://guglielmogattiglio.com/blog/bayescart-python-package/>`_.

For theoretical background on Bayesian CART, and the specific tempering strategies implemented in this package, check this `detailed blog series <https://guglielmogattiglio.com/blog/bayesian-classification-and-regression-trees-theoretical-series>`_.

For an example on how to use this package, see this `tutorial notebook <https://guglielmogattiglio.com/blog/using-bayescart-to-solve-cgm98>`_.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules