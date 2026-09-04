# Configuration file for the Sphinx documentation builder.
# underPINN-v2605 documentation
#
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "underPINN"
copyright = "2026, Kumar Prashant, Senthilkumar Lohith, Ranjan Rajesh"
author = "Kumar Prashant, Senthilkumar Lohith, Ranjan Rajesh"
#release = "v2605"
#version = "v2605"

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "attrs_inline",
    "tasklist",
]

source_suffix = {
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "underPINN"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "underPINN-logo.png"

html_context = {
    "default_mode": "dark",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

html_theme_options = {
    "repository_url": "https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "path_to_docs": "docs", # Updated to match your flat folder structure
    "repository_branch": "main",
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "logo": {
        "text": "underPINN",
        "image_light": "underPINN-logo-light-mode.png",  
        "image_dark": "underPINN-logo-dark-mode.png",    
    },
}

pygments_style = "monokai"
pygments_dark_style = "monokai"