# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add the parent directory of your package to the Python path
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

project = 'BayesCART'
copyright = '2025, Guglielmo Gattiglio'
author = 'Guglielmo Gattiglio'
html_title = "BayesCART Documentation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supports Google & NumPy docstrings
    'sphinx.ext.viewcode',  # Adds source code links
    'sphinx.ext.autodoc.typehints',  # Show type hints in docs
]

extensions.append('sphinx_autodoc_typehints')

autodoc_inherit_docstrings = True  # Enable docstring inheritance
autoclass_content = 'both'
napoleon_google_docstring = True  # Enable Google-style docstrings
napoleon_numpy_docstring = True  # Enable NumPy-style docstrings


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",  # Ensure only __init__ is included
    "show-inheritance": True,
}